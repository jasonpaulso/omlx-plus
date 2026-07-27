// SPDX-License-Identifier: Apache-2.0
//
// The cluster panel, as its own Alpine island.
//
// Deliberately not part of dashboard(): clustering is off by default and the
// panel renders nothing at all when it is off, so its state has no business
// living in the dashboard's. Keeping it separate also means the panel can be
// dropped in and out without touching dashboard.js.

function clusterPanel() {
    return {
        loaded: false,
        status: { enabled: false, formed: false, nodes: [], peers: [], blockers: [] },
        preflight: null,
        showPreflight: false,
        busy: false,
        error: '',
        timer: null,

        async init() {
            await this.refresh();
            // Formation and teardown happen out of band (a model load forms the
            // cluster), so the panel polls rather than waiting to be told.
            this.timer = setInterval(() => this.refresh(), 5000);
        },

        destroy() {
            if (this.timer) clearInterval(this.timer);
        },

        async refresh() {
            try {
                const response = await fetch('/admin/api/cluster/status');
                if (!response.ok) return;
                this.status = await response.json();
                this.loaded = true;
            } catch (e) {
                // A poll that fails is not worth a banner; the next one is 5s away.
            }
        },

        async loadPreflight() {
            this.showPreflight = !this.showPreflight;
            if (!this.showPreflight || this.preflight) return;
            try {
                const response = await fetch('/admin/api/cluster/preflight');
                this.preflight = await response.json();
            } catch (e) {
                this.error = String(e);
            }
        },

        async teardown() {
            this.busy = true;
            this.error = '';
            try {
                const response = await fetch('/admin/api/cluster/teardown', {
                    method: 'POST',
                });
                if (!response.ok) this.error = await response.text();
                await this.refresh();
            } catch (e) {
                this.error = String(e);
            } finally {
                this.busy = false;
            }
        },

        get backendLabel() {
            const backend = this.status.backend;
            if (!backend) return '';
            return backend === 'ring' ? 'TCP ring' : backend;
        },

        // Ranks are shown in rank order because for jaccl-ring that order is
        // the physical cabling, not a presentation choice.
        get ranks() {
            return (this.status.nodes || []).slice().sort((a, b) => a.rank - b.rank);
        },

        get unusedPeers() {
            const joined = new Set((this.status.nodes || []).map((n) => n.node_id));
            return (this.status.peers || []).filter((p) => !joined.has(p.node_id));
        },
    };
}
