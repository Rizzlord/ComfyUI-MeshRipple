import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "ComfyUI.MeshRipple",
    async nodeCreated(node) {
        if (node.comfyClass !== "MeshRippleGenerator") return;

        node.mesh_yaw = 0.5;
        node.mesh_pitch = 0.3;
        node.mesh_zoom = 1.0;
        node.mesh_data = null;

        const widget = {
            type: "mesh_preview",
            name: "mesh_preview",
            value: null,
            height: 220,
            draw(ctx, nd, widget_width, y, widget_height) {
                const actual_height = Math.max(100, nd.size[1] - y - 16);
                widget.height = actual_height;

                ctx.fillStyle = "#121214";
                ctx.fillRect(0, y, widget_width, actual_height);
                ctx.strokeStyle = "#2e2f38";
                ctx.lineWidth = 1.5;
                ctx.strokeRect(0, y, widget_width, actual_height);

                const mesh = nd.mesh_data;
                if (!mesh || !mesh.vertices || mesh.vertices.length === 0) {
                    ctx.fillStyle = "#6b6d7a";
                    ctx.font = "11px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("Waiting for generation...", widget_width / 2, y + actual_height / 2);
                    return;
                }

                let minX = Infinity, minY = Infinity, minZ = Infinity;
                let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
                for (let i = 0; i < mesh.vertices.length; i++) {
                    const v = mesh.vertices[i];
                    if (v[0] < minX) minX = v[0];
                    if (v[1] < minY) minY = v[1];
                    if (v[2] < minZ) minZ = v[2];
                    if (v[0] > maxX) maxX = v[0];
                    if (v[1] > maxY) maxY = v[1];
                    if (v[2] > maxZ) maxZ = v[2];
                }

                const cx = (minX + maxX) / 2;
                const cy = (minY + maxY) / 2;
                const cz = (minZ + maxZ) / 2;

                const dx = maxX - minX || 0.001;
                const dy = maxY - minY || 0.001;
                const dz = maxZ - minZ || 0.001;
                const maxDim = Math.max(dx, dy, dz);

                const cosY = Math.cos(nd.mesh_yaw);
                const sinY = Math.sin(nd.mesh_yaw);
                const cosP = Math.cos(nd.mesh_pitch);
                const sinP = Math.sin(nd.mesh_pitch);

                const projected = [];
                nd.mesh_zoom = nd.mesh_zoom || 1.0;
                const scale = 0.8 * Math.min(widget_width, actual_height) / maxDim * nd.mesh_zoom;
                const centerX = widget_width / 2;
                const centerY = y + actual_height / 2;

                for (let i = 0; i < mesh.vertices.length; i++) {
                    const v = mesh.vertices[i];
                    const x = v[0] - cx;
                    const y_val = v[1] - cy;
                    const z = v[2] - cz;

                    const rx = x * cosY - z * sinY;
                    const rz = x * sinY + z * cosY;

                    const ry = y_val * cosP - rz * sinP;
                    const rz2 = y_val * sinP + rz * cosP;

                    const dist = 2.0;
                    const factor = 1.0 / (dist - rz2 / maxDim);
                    const sx = centerX + rx * scale * factor;
                    const sy = centerY - ry * scale * factor;

                    projected.push({ x: sx, y: sy, z: rz2 });
                }

                const sortedFaces = [];
                for (let i = 0; i < mesh.faces.length; i++) {
                    const f = mesh.faces[i];
                    const p0 = projected[f[0]];
                    const p1 = projected[f[1]];
                    const p2 = projected[f[2]];
                    if (p0 && p1 && p2) {
                        const avgZ = (p0.z + p1.z + p2.z) / 3;
                        sortedFaces.push({ f, avgZ });
                    }
                }
                sortedFaces.sort((a, b) => a.avgZ - b.avgZ);

                for (let i = 0; i < sortedFaces.length; i++) {
                    const { f } = sortedFaces[i];
                    const p0 = projected[f[0]];
                    const p1 = projected[f[1]];
                    const p2 = projected[f[2]];

                    const v0 = mesh.vertices[f[0]];
                    const v1 = mesh.vertices[f[1]];
                    const v2 = mesh.vertices[f[2]];

                    const ux = v1[0] - v0[0];
                    const uy = v1[1] - v0[1];
                    const uz = v1[2] - v0[2];
                    const vx = v2[0] - v0[0];
                    const vy = v2[1] - v0[1];
                    const vz = v2[2] - v0[2];
                    const nx = uy * vz - uz * vy;
                    const ny = uz * vx - ux * vz;
                    const nz = ux * vy - uy * vx;
                    const len = Math.hypot(nx, ny, nz) || 1;
                    const dnz = nz / len;

                    const s_ux = p1.x - p0.x;
                    const s_uy = p1.y - p0.y;
                    const s_vx = p2.x - p0.x;
                    const s_vy = p2.y - p0.y;
                    const s_nz = s_ux * s_vy - s_uy * s_vx;

                    const dot = s_nz > 0 ? 1 : -1;
                    const normal_z = dnz * dot;
                    const intensity = Math.max(0.15, Math.min(1.0, 0.4 + 0.6 * normal_z));

                    const r = Math.round(30 * intensity);
                    const g = Math.round(130 * intensity);
                    const b = Math.round(230 * intensity);

                    ctx.beginPath();
                    ctx.moveTo(p0.x, p0.y);
                    ctx.lineTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.closePath();

                    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                    ctx.fill();

                    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
                    ctx.stroke();
                }
            },
            computeSize() {
                return [220, widget.height || 220];
            },
            mouse(event, pos, nd) {
                if (event.type === "mousedown" || event.type === "pointerdown") {
                    this.mouse_dragging = true;
                    this.last_mouse_pos = [pos[0], pos[1]];
                    return true;
                } else if (event.type === "mousemove" || event.type === "pointermove") {
                    if (this.mouse_dragging) {
                        const dx = pos[0] - this.last_mouse_pos[0];
                        const dy = pos[1] - this.last_mouse_pos[1];
                        nd.mesh_yaw = (nd.mesh_yaw || 0) - dx * 0.01;
                        nd.mesh_pitch = (nd.mesh_pitch || 0) + dy * 0.01;
                        this.last_mouse_pos = [pos[0], pos[1]];
                        nd.setDirtyCanvas(true, true);
                        return true;
                    }
                } else if (event.type === "mouseup" || event.type === "pointerup" || event.type === "pointerout") {
                    this.mouse_dragging = false;
                    return true;
                }
                return false;
            }
        };

        node.addCustomWidget(widget);
        node.setSize([node.size[0] || 240, (node.size[1] || 260) + 240]);

        node.onMouseWheel = function(event, pos, canvas) {
            const w = this.widgets.find(item => item.name === "mesh_preview");
            if (w && pos[1] >= w.y && pos[1] <= w.y + w.height) {
                const zoom_factor = event.deltaY < 0 ? 1.1 : 0.9;
                this.mesh_zoom = Math.max(0.1, Math.min(10.0, (this.mesh_zoom || 1.0) * zoom_factor));
                this.setDirtyCanvas(true, true);
                return true;
            }
        };
    }
});

api.addEventListener("meshripple_preview", (event) => {
    const data = event.detail;
    const node = app.graph.getNodeById(data.node_id);
    if (node) {
        node.mesh_data = data;
        node.setDirtyCanvas(true, true);
    }
});
