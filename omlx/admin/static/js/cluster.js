// SPDX-License-Identifier: Apache-2.0
//
// The cluster tab, as its own Alpine island.
//
// Deliberately not part of dashboard(): this page has to work on a machine
// that has never had clustering enabled - it is what enables it - so it owns
// its own loading, its own model list and its own polling, and can be dropped
// in or out without touching dashboard.js.

function clusterPanel() {
    return {
        loaded: false,
        status: { enabled: false, formed: false, nodes: [], peers: [], blockers: [] },
        config: {
            enabled: false,
            cluster_key: '',
            backend: 'auto',
            model: '',
            pipeline: false,
            max_batch_size: 8,
            discovery_interval_seconds: 5,
        },
        models: [],
        modelOptions: [],
        peerChecks: {},
        repoId: '',
        checkedModel: '',
        dismissedMissing: false,
        startingDownload: false,
        downloads: [],
        downloadTimer: null,
        forming: false,
        preflight: null,
        showPreflight: false,
        revealKey: false,
        keyCopied: false,
        busy: false,
        saving: false,
        checking: false,
        saved: false,
        error: '',
        timer: null,

        async init() {
            await Promise.all([this.refresh(), this.loadConfig(), this.loadModels()]);
            this.rebuildModelOptions();
            // Formation happens out of band - loading the sharded model forms
            // the cluster - so the page polls rather than waiting to be told.
            this.$watch('mainTab', () => this.syncTimer());
            this.$watch('config.model', () => {
                this.peerChecks = {};
                this.dismissedMissing = false;
                if (this.config.model && this.status.peers?.length) this.checkPeers();
            });
            await this.pollDownloads();
            if (this.downloads.length) this.startDownloadTimer();
            this.syncTimer();
        },

        destroy() {
            this.stopTimer();
            this.stopDownloadTimer();
        },

        // `mainTab` is read from the dashboard scope this island sits inside.
        // If it is ever not there, poll anyway rather than silently showing a
        // frozen page.
        get isActive() {
            return this.mainTab === undefined || this.mainTab === 'cluster';
        },

        syncTimer() {
            this.stopTimer();
            if (!this.isActive) return;
            this.timer = setInterval(() => this.refresh(), 5000);
        },

        stopTimer() {
            if (this.timer) clearInterval(this.timer);
            this.timer = null;
        },

        async refresh(explicit = false) {
            if (explicit) this.busy = true;
            try {
                const response = await fetch('/admin/api/cluster/status');
                if (!response.ok) return;
                this.status = await response.json();
                this.loaded = true;
            } catch (e) {
                // A poll that fails is not worth a banner; the next one is 5s away.
                if (explicit) this.error = String(e);
            } finally {
                if (explicit) this.busy = false;
            }
        },

        async loadConfig() {
            try {
                const response = await fetch('/admin/api/cluster/config');
                if (!response.ok) return;
                this.config = { ...this.config, ...(await response.json()) };
            } catch (e) {
                this.error = String(e);
            }
        },

        async loadModels() {
            try {
                const response = await fetch('/admin/api/cluster/candidates');
                if (!response.ok) return;
                this.models = (await response.json()).candidates || [];
            } catch (e) {
                // The picker degrades to whatever is already configured.
            }
        },

        async saveConfig() {
            this.saving = true;
            this.error = '';
            this.saved = false;
            try {
                const response = await fetch('/admin/api/cluster/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        enabled: this.config.enabled,
                        cluster_key: this.config.cluster_key,
                        backend: this.config.backend,
                        model: this.config.model,
                        pipeline: this.config.pipeline,
                        max_batch_size: this.config.max_batch_size,
                        discovery_interval_seconds: this.config.discovery_interval_seconds,
                    }),
                });
                if (!response.ok) {
                    this.error = await this.errorText(response);
                    return;
                }
                const data = await response.json();
                this.config = { ...this.config, ...(data.config || {}) };
                this.rebuildModelOptions();
                this.saved = true;
                // A changed key invalidates every peer verdict on the page.
                this.peerChecks = {};
                await this.refresh();
            } catch (e) {
                this.error = String(e);
            } finally {
                this.saving = false;
            }
        },

        async generateKey() {
            this.error = '';
            try {
                const response = await fetch('/admin/api/cluster/key', { method: 'POST' });
                if (!response.ok) {
                    this.error = await this.errorText(response);
                    return;
                }
                const data = await response.json();
                this.config.cluster_key = data.cluster_key;
                this.revealKey = true;
            } catch (e) {
                this.error = String(e);
            }
        },

        async copyKey() {
            try {
                await navigator.clipboard.writeText(this.config.cluster_key);
                this.keyCopied = true;
                setTimeout(() => { this.keyCopied = false; }, 2000);
            } catch (e) {
                // Clipboard access needs a secure context; showing the field
                // and letting the operator select it is the fallback.
                this.revealKey = true;
            }
        },

        async checkPeers() {
            if (!this.config.model) return;
            this.checking = true;
            this.error = '';
            this.dismissedMissing = false;
            try {
                const response = await fetch('/admin/api/cluster/peers/check', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: this.config.model }),
                });
                if (!response.ok) {
                    this.error = await this.errorText(response);
                    return;
                }
                const data = await response.json();
                const checks = {};
                for (const peer of data.peers || []) checks[peer.node_id] = peer;
                this.peerChecks = checks;
                // Empty when the model has no HuggingFace origin, which is the
                // difference between "we can fix this for you" and "copy it
                // across yourself".
                this.repoId = data.repo_id || '';
                this.checkedModel = data.model || '';
            } catch (e) {
                this.error = String(e);
            } finally {
                this.checking = false;
            }
        },

        // Peers that answered and do not have the sharded model. A peer we
        // could not reach is a different problem and gets its own error line.
        get peersMissingModel() {
            return Object.values(this.peerChecks)
                .filter((c) => c.ok && c.has_model === false);
        },

        get missingPeerNames() {
            const byId = {};
            for (const p of this.status.peers || []) byId[p.node_id] = p;
            return this.peersMissingModel
                .map((c) => (byId[c.node_id]?.hostname) || c.node_id)
                .join(', ');
        },

        async downloadOnPeers() {
            this.startingDownload = true;
            this.error = '';
            try {
                const response = await fetch('/admin/api/cluster/peers/download', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.config.model,
                        node_ids: this.peersMissingModel.map((c) => c.node_id),
                    }),
                });
                if (!response.ok) {
                    this.error = await this.errorText(response);
                    return;
                }
                const data = await response.json();
                const failed = (data.peers || []).filter((p) => !p.ok);
                if (failed.length) {
                    this.error = failed.map((p) => `${p.node_id}: ${p.error}`).join('; ');
                }
                this.dismissedMissing = true;
                await this.pollDownloads();
                this.startDownloadTimer();
            } catch (e) {
                this.error = String(e);
            } finally {
                this.startingDownload = false;
            }
        },

        startDownloadTimer() {
            this.stopDownloadTimer();
            this.downloadTimer = setInterval(() => this.pollDownloads(), 2000);
        },

        stopDownloadTimer() {
            if (this.downloadTimer) clearInterval(this.downloadTimer);
            this.downloadTimer = null;
        },

        async pollDownloads() {
            try {
                const response = await fetch('/admin/api/cluster/peers/downloads');
                if (!response.ok) return;
                this.downloads = (await response.json()).downloads || [];
            } catch (e) {
                return;
            }
            if (!this.downloads.length) {
                this.stopDownloadTimer();
                return;
            }
            const states = this.downloads.map((d) => d.task?.status || 'pending');
            if (states.some((s) => s === 'pending' || s === 'downloading')) return;

            this.stopDownloadTimer();
            // Every download settled. Only a clean sweep earns an automatic
            // formation; a failure stays on screen for the operator to read.
            if (states.every((s) => s === 'completed')) {
                await this.checkPeers();
                if (!this.peersMissingModel.length) await this.formCluster();
            }
        },

        async formCluster() {
            this.forming = true;
            this.error = '';
            try {
                const response = await fetch('/admin/api/cluster/form', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: this.config.model }),
                });
                if (!response.ok) {
                    this.error = await this.errorText(response);
                    return;
                }
                await fetch('/admin/api/cluster/peers/downloads/clear', { method: 'POST' });
                this.downloads = [];
                await this.refresh();
            } catch (e) {
                this.error = String(e);
            } finally {
                this.forming = false;
            }
        },

        async togglePreflight() {
            this.showPreflight = !this.showPreflight;
            if (!this.showPreflight || this.preflight) return;
            try {
                const response = await fetch('/admin/api/cluster/preflight');
                if (response.ok) this.preflight = await response.json();
            } catch (e) {
                this.error = String(e);
            }
        },

        async teardown() {
            this.busy = true;
            this.error = '';
            try {
                const response = await fetch('/admin/api/cluster/teardown', { method: 'POST' });
                if (!response.ok) this.error = await this.errorText(response);
                await this.refresh();
            } catch (e) {
                this.error = String(e);
            } finally {
                this.busy = false;
            }
        },

        async errorText(response) {
            try {
                const data = await response.json();
                return data.detail || JSON.stringify(data);
            } catch (e) {
                return `${response.status} ${response.statusText}`;
            }
        },

        peerLabel(nodeId) {
            const peer = (this.status.peers || []).find((p) => p.node_id === nodeId);
            return peer?.hostname || this.hostLabel(peer?.host) || nodeId;
        },

        transferModeLabel(dl) {
            return dl.mode === 'transfer'
                ? window.t('cluster.download.mode_transfer')
                : `${window.t('cluster.download.mode_download')} · ${dl.source || ''}`;
        },

        downloadLabel(dl) {
            if (dl.error) return window.t('cluster.download.failed');
            const task = dl.task || {};
            if (task.status === 'completed') return window.t('cluster.download.done');
            if (task.status === 'failed') return window.t('cluster.download.failed');
            if (task.status === 'downloading') return `${Math.round(task.progress || 0)}%`;
            return window.t('cluster.download.waiting');
        },

        downloadTone(dl) {
            const status = dl.error ? 'failed' : (dl.task?.status || 'pending');
            if (status === 'completed') return 'text-green-700';
            if (status === 'failed') return 'text-red-600';
            return 'text-neutral-500';
        },

        // Bonjour hands back either a `.local` name or a bare address. Only
        // the former is worth showing as a machine's name.
        hostLabel(host) {
            if (!host || !host.endsWith('.local')) return '';
            return host.slice(0, -'.local'.length);
        },

        // Rebuilt by hand rather than exposed as a getter. An `x-for` over a
        // getter binds its effect on first evaluation - which happens before
        // the model list has been fetched - and never re-runs, leaving the
        // picker permanently empty while the getter itself returns the right
        // answer to anything that asks. A plain assigned array is also what
        // every other dynamic select in this dashboard uses.
        rebuildModelOptions() {
            const options = this.models.map((m) => ({
                value: m.id,
                label: m.size_formatted ? `${m.display_name} · ${m.size_formatted}` : m.display_name,
                // An ineligible model stays selectable. Being told a model
                // cannot be sharded, and why, beats it silently not being in
                // the list - which is how one gets picked that could never
                // have formed.
                disabled: !m.eligible,
                reason: m.reason || '',
            }));
            if (this.config.model && !this.models.some((m) => m.id === this.config.model)) {
                options.unshift({ value: this.config.model, label: this.config.model, disabled: false, reason: '' });
            }
            this.modelOptions = options;
        },

        get selectedCandidate() {
            return this.models.find((m) => m.id === this.config.model) || null;
        },

        // Why the currently selected model cannot be clustered, if it cannot.
        get modelWarning() {
            const chosen = this.selectedCandidate;
            return chosen && !chosen.eligible ? chosen.reason : '';
        },

        // Driven by x-effect on the select. Reading `modelOptions` and
        // `config.model` here is what subscribes the effect to both, so the
        // picker refills when the model list arrives and keeps the saved
        // selection after a save replaces the config object.
        syncModelOptions(select) {
            const options = this.modelOptions;
            const selected = this.config.model || '';
            // Everything after the placeholder is ours to replace.
            while (select.options.length > 1) select.remove(1);
            for (const option of options) {
                const el = document.createElement('option');
                el.value = option.value;
                el.textContent = option.reason
                    ? `${option.label} — ${option.reason}`
                    : option.label;
                el.disabled = !!option.disabled;
                select.appendChild(el);
            }
            select.value = selected;
        },

        get backendLabel() {
            const backend = this.status.backend || (this.status.enabled ? this.config.backend : '');
            if (!backend) return '';
            if (backend === 'ring') return 'TCP ring';
            if (backend === 'auto') return window.t('cluster.config.backend_auto');
            return backend;
        },

        // Ranks this node holds for a cluster some other Mac leads. A follower
        // has no manager, so `formed` is false here and without this the page
        // would read as an idle, unconfigured node while it is in fact busy
        // holding a shard of someone else's model.
        get followerRanks() {
            return this.status.follower_ranks || [];
        },

        get stateLabel() {
            if (!this.loaded) return '';
            if (!this.status.enabled) return window.t('cluster.state.off');
            if (this.status.busy) return window.t('cluster.state.serving');
            if (this.status.formed) return window.t('cluster.state.formed');
            if (this.followerRanks.length) return window.t('cluster.state.holding_ranks');
            if (this.joinablePeers.length) return window.t('cluster.state.ready');
            return window.t('cluster.state.searching');
        },

        get stateClass() {
            if (!this.status.enabled) return 'bg-neutral-100 text-neutral-500';
            if (this.status.busy) return 'bg-amber-50 text-amber-700';
            if (this.status.formed || this.followerRanks.length) return 'bg-green-50 text-green-700';
            return 'bg-neutral-100 text-neutral-500';
        },

        // Peers that would actually join: a mismatched key never will, and
        // saying so beats leaving them out of the count with no explanation.
        get joinablePeers() {
            return (this.status.peers || []).filter((p) => p.key_match !== false);
        },

        get fleetGb() {
            const local = this.status.local?.ram_gb || 0;
            return this.joinablePeers.reduce((sum, p) => sum + (p.ram_gb || 0), local);
        },

        get fleetMemoryLabel() {
            return this.fleetGb ? `${this.fleetGb} GB` : '—';
        },

        // The engine pool deliberately skips this daemon's memory ceiling for
        // a cluster model - the weights are never in this process - so this
        // figure is what replaces the refusal the operator no longer gets.
        // Display only: it is a sanity check, not an admission decision.
        get modelSizeLabel() {
            const chosen = this.models.find(
                (m) => m.id === this.status.model || m.id === this.config.model
            );
            if (!chosen || !chosen.size_formatted) return '';
            return this.fleetGb
                ? `${chosen.size_formatted} of ${this.fleetGb} GB`
                : chosen.size_formatted;
        },

        get fleetNodesLabel() {
            const nodes = this.joinablePeers.length + (this.status.local?.node_id ? 1 : 0);
            if (nodes <= 1) return window.t('cluster.card.fleet_memory_hint_one');
            return window.t('cluster.card.fleet_memory_hint').replace('{nodes}', nodes);
        },

        // This Mac first, then every peer. Ranks come from the formed cluster
        // when there is one; rank order is the cabling for jaccl-ring, not a
        // presentation choice.
        get nodeRows() {
            const ranks = new Map((this.status.nodes || []).map((n) => [n.node_id, n]));
            const local = this.status.local || {};
            const rows = [];
            if (local.node_id) {
                rows.push({
                    node_id: local.node_id,
                    // The node id is a hardware UUID. Only ever show it when a
                    // peer is too old to advertise a name.
                    label: local.hostname || local.node_id,
                    chip: local.chip,
                    ram_gb: local.ram_gb,
                    version: local.version,
                    host: '',
                    port: local.port,
                    is_local: true,
                    key_match: null,
                    has_model: null,
                    rank: ranks.get(local.node_id)?.rank ?? this.followerRanks[0] ?? null,
                });
            }
            for (const peer of this.status.peers || []) {
                const check = this.peerChecks[peer.node_id];
                rows.push({
                    ...peer,
                    // A peer too old to advertise a name still resolved to
                    // something readable through Bonjour; the hardware UUID is
                    // the last resort, not the second one.
                    label: peer.hostname || this.hostLabel(peer.host) || peer.node_id,
                    is_local: false,
                    has_model: check ? check.has_model : null,
                    rank: ranks.get(peer.node_id)?.rank ?? null,
                });
            }
            return rows.sort((a, b) => {
                if (a.rank === b.rank) return a.is_local ? -1 : 1;
                if (a.rank === null) return 1;
                if (b.rank === null) return -1;
                return a.rank - b.rank;
            });
        },

        get peerRows() {
            return this.status.peers || [];
        },

        get saveHint() {
            if (this.saved) return window.t('cluster.config.saved');
            if (this.followerRanks.length) return window.t('cluster.follower_hint');
            if (this.status.formed && this.status.busy) return window.t('cluster.config.busy');
            return window.t('cluster.config.restart_hint');
        },
    };
}
