// The 3D view: renderer, camera, sky, lights, the site's meshes, picking and highlighting.
//
// Axes are glTF's: x east, y up, z south (north is -z). One unit is one metre.

import {
  Box3,
  BufferAttribute,
  BufferGeometry,
  CircleGeometry,
  Color,
  DirectionalLight,
  EdgesGeometry,
  Fog,
  Group,
  LineBasicMaterial,
  LineSegments,
  MathUtils,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  NeutralToneMapping,
  PCFShadowMap,
  PerspectiveCamera,
  PMREMGenerator,
  Raycaster,
  Scene,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { Sky } from "three/addons/objects/Sky.js";
import { FeatureIndex } from "./features";

// Display-only light: a fixed sun from the south-east, while the camera starts in the
// south-west, so shadows fall where the camera can see them. It is not the real sun of the
// site; sun and shade are simulated in a later phase.
const DISPLAY_SUN_ELEVATION_DEG = 40;
const DISPLAY_SUN_AZIMUTH_DEG = 135; // clockwise from north

const HIGHLIGHT = new Color("#ffb000");

export interface Hit {
  group: string;
  feature: number;
  id: string;
}

interface SiteGroup {
  mesh: Mesh;
  ids: string[];
  index: FeatureIndex;
}

/** Direction towards the sun in glTF axes, from elevation and azimuth (clockwise from north). */
export function sunDirection(elevationDeg: number, azimuthDeg: number): Vector3 {
  const el = MathUtils.degToRad(elevationDeg);
  const az = MathUtils.degToRad(azimuthDeg);
  // North is -z and east is +x.
  return new Vector3(Math.cos(el) * Math.sin(az), Math.sin(el), -Math.cos(el) * Math.cos(az));
}

export class Viewer {
  readonly renderer: WebGLRenderer;
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(40, 1, 1, 30000);
  readonly controls: OrbitControls;
  readonly groups = new Map<string, SiteGroup>();
  /** Called after every frame, for overlays such as the compass. */
  onRender: (() => void) | null = null;

  private readonly sun = new DirectionalLight("#fff3e0", 2.6);
  private readonly sunDir = sunDirection(DISPLAY_SUN_ELEVATION_DEG, DISPLAY_SUN_AZIMUTH_DEG);
  private readonly raycaster = new Raycaster();
  private highlight: Group | null = null;
  private frameRequested = false;
  private leftInset = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.renderer = new WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFShadowMap;
    // Khronos PBR Neutral tone mapping keeps base colours close to their sRGB values, which
    // matters here because colour carries meaning (where a height came from).
    this.renderer.toneMapping = NeutralToneMapping;
    this.renderer.toneMappingExposure = 1.0;

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = MathUtils.degToRad(86); // stay above the ground
    this.controls.addEventListener("change", () => this.requestRender());

    this.addSky();
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(4096, 4096);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.6;
    this.scene.add(this.sun, this.sun.target);

    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  private addSky(): void {
    const sky = new Sky();
    sky.scale.setScalar(20000);
    const u = sky.material.uniforms;
    u["turbidity"]!.value = 2.5;
    u["rayleigh"]!.value = 1.1;
    u["mieCoefficient"]!.value = 0.004;
    u["mieDirectionalG"]!.value = 0.8;
    u["cloudCoverage"]!.value = 0.3;
    (u["sunPosition"]!.value as Vector3).copy(this.sunDir);
    this.scene.add(sky);

    // Soft light from the whole sky (image-based lighting): render the sky once into an
    // environment map that every material reflects a little of.
    const envScene = new Scene();
    const envSky = new Sky();
    envSky.scale.setScalar(50);
    envSky.material.uniforms = sky.material.uniforms;
    envSky.material.uniforms["showSunDisc"]!.value = 0;
    envScene.add(envSky);
    const pmrem = new PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(envScene, 0, 0.1, 100).texture;
    this.scene.environmentIntensity = 0.4;
    envSky.material.uniforms["showSunDisc"]!.value = 1;
    pmrem.dispose();
  }

  /** Load the site's scene.glb and frame the camera on it. */
  async load(url: string): Promise<void> {
    const gltf = await new GLTFLoader().loadAsync(url);
    const meshes: Mesh[] = [];
    gltf.scene.traverse((o) => {
      if ((o as Mesh).isMesh) meshes.push(o as Mesh);
    });
    for (const mesh of meshes) {
      const name = String(mesh.userData["group"] ?? mesh.name);
      const ids = (mesh.userData["feature_ids"] as string[] | undefined) ?? [];
      const geometry = mesh.geometry;
      const attr = geometry.getAttribute("_feature_id");
      const featureOfVertex = attr
        ? Float32Array.from({ length: attr.count }, (_, i) => attr.getX(i))
        : new Float32Array(geometry.getAttribute("position").count);
      const triangles = (geometry.index?.count ?? geometry.getAttribute("position").count) / 3;
      const index = new FeatureIndex(
        featureOfVertex,
        geometry.index ? (geometry.index.array as ArrayLike<number>) : null,
        triangles,
      );
      this.groups.set(name, { mesh, ids, index });
      this.style(name, mesh);
    }
    this.scene.add(gltf.scene);
    this.setUpSurroundings();
    this.frame();
  }

  private style(name: string, mesh: Mesh): void {
    const material = mesh.material as MeshStandardMaterial;
    mesh.receiveShadow = true;
    mesh.castShadow = name.startsWith("buildings") || name.startsWith("tree");
    if (name === "roads" || name === "footways") {
      // Drawn a few centimetres above the ground; push them forward in depth so they never
      // flicker against it when seen from far away.
      material.polygonOffset = true;
      material.polygonOffsetFactor = name === "roads" ? -1 : -2;
      material.polygonOffsetUnits = name === "roads" ? -1 : -2;
    }
    if (name === "ground") {
      // The ground is smooth hills, not facets: average the normals at shared corners.
      mesh.geometry.computeVertexNormals();
      material.flatShading = false;
      material.needsUpdate = true;
    }
    if (name === "site_outline") {
      material.polygonOffset = true;
      material.polygonOffsetFactor = -3;
      material.polygonOffsetUnits = -3;
      mesh.castShadow = false;
    }
    if (name.startsWith("buildings")) {
      const edges = new LineSegments(
        new EdgesGeometry(mesh.geometry, 30),
        new LineBasicMaterial({ color: "#20262c", transparent: true, opacity: 0.32 }),
      );
      edges.raycast = () => {}; // edges are decoration; never pick them
      mesh.add(edges);
    }
  }

  /** The simulated square (marked by the site outline), from its lowest ground up to its
   *  highest feature. The ground mesh reaches further: it shows the surroundings. */
  private siteBox(): Box3 {
    const box = new Box3();
    const outline = this.groups.get("site_outline")?.mesh;
    const ground = this.groups.get("ground")?.mesh;
    if (outline) box.expandByObject(outline);
    else if (ground) box.expandByObject(ground);
    else for (const g of this.groups.values()) box.expandByObject(g.mesh);
    if (box.isEmpty()) box.set(new Vector3(-100, 0, -100), new Vector3(100, 50, 100));
    const inSquare = (b: Box3) =>
      b.max.x >= box.min.x && b.min.x <= box.max.x && b.max.z >= box.min.z && b.min.z <= box.max.z;
    let top = box.max.y;
    for (const [name, g] of this.groups) {
      if (name === "ground") continue;
      const b = new Box3().setFromObject(g.mesh);
      if (inSquare(b)) top = Math.max(top, b.max.y);
    }
    box.max.y = Math.max(top, box.min.y + 1);
    return box;
  }

  /** The lowest point of the drawn ground (site and surroundings). */
  private groundFloor(): number {
    const ground = this.groups.get("ground")?.mesh;
    return ground ? new Box3().setFromObject(ground).min.y : 0;
  }

  /** Shadows, the ground around the site and the haze, sized to the site. Once, after loading. */
  private setUpSurroundings(): void {
    const box = this.siteBox();
    const centre = box.getCenter(new Vector3());
    const radius = Math.max(box.max.x - box.min.x, box.max.z - box.min.z) / 2;

    const reach = radius * 1.5 + (box.max.y - box.min.y);
    this.sun.position.copy(centre).addScaledVector(this.sunDir, reach * 2);
    this.sun.target.position.copy(centre);
    const cam = this.sun.shadow.camera;
    cam.left = cam.bottom = -reach;
    cam.right = cam.top = reach;
    cam.near = 1;
    cam.far = reach * 4;
    cam.updateProjectionMatrix();

    // Ground beyond the site, fading into the haze, so the model does not float in the sky.
    const outside = new Mesh(
      // Large enough that its edge sits almost on the horizon, hidden in the haze.
      new CircleGeometry(radius * 60, 96).rotateX(-Math.PI / 2),
      new MeshStandardMaterial({ color: "#7d8a77", roughness: 1 }),
    );
    // Just below the lowest drawn ground, so it never cuts through the hills.
    outside.position.set(centre.x, this.groundFloor() - 1, centre.z);
    outside.receiveShadow = true;
    outside.raycast = () => {};
    this.scene.add(outside);
    this.scene.fog = new Fog("#d3dee5", radius * 2.5, radius * 16);
    this.controls.maxDistance = radius * 8;
    this.controls.minDistance = 5;
  }

  /**
   * Look at the whole site from the south-west: move the camera back along that direction
   * until every corner of the site's box, tallest building included, is on screen and clear
   * of the panel.
   */
  frame(): void {
    const box = this.siteBox();
    const target = box.getCenter(new Vector3());
    target.y = box.min.y + (box.max.y - box.min.y) * 0.2;
    const direction = new Vector3(-1.45, 1.05, 1.75).normalize();
    const corners = [0, 1, 2, 3, 4, 5, 6, 7].map(
      (k) =>
        new Vector3(
          k & 1 ? box.max.x : box.min.x,
          k & 2 ? box.max.y : box.min.y,
          k & 4 ? box.max.z : box.min.z,
        ),
    );
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    const margin = 24; // pixels
    const fits = (distance: number) => {
      this.camera.position.copy(target).addScaledVector(direction, distance);
      this.camera.lookAt(target);
      this.camera.updateMatrixWorld();
      return corners.every((c) => {
        const p = c.clone().project(this.camera);
        const x = ((p.x + 1) / 2) * w;
        const y = ((1 - p.y) / 2) * h;
        return (
          p.z < 1 && x > this.leftInset + margin && x < w - margin && y > margin && y < h - margin
        );
      });
    };
    let near = 1;
    let far = 1;
    while (!fits(far) && far < 1e6) far *= 2;
    for (let i = 0; i < 40; i++) {
      const mid = (near + far) / 2;
      if (fits(mid)) far = mid;
      else near = mid;
    }
    fits(far);
    this.controls.target.copy(target);
    this.controls.update();
    this.requestRender();
  }

  setVisible(group: string, visible: boolean): void {
    const g = this.groups.get(group);
    if (!g) return;
    g.mesh.visible = visible;
    if (!visible && this.highlight?.userData["group"] === group) this.select(null);
    this.requestRender();
  }

  /** The feature under a point on the screen (client pixels), or null. */
  pick(clientX: number, clientY: number): Hit | null {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.camera);
    const targets = [...this.groups.entries()]
      .filter(([name, g]) => name !== "ground" && g.mesh.visible)
      .map(([, g]) => g.mesh);
    for (const hit of this.raycaster.intersectObjects(targets, false)) {
      if (hit.faceIndex == null) continue;
      const name = String(hit.object.userData["group"] ?? hit.object.name);
      const g = this.groups.get(name);
      if (!g) continue;
      const feature = g.index.featureOfTriangle(hit.faceIndex);
      const id = g.ids[feature];
      if (id !== undefined) return { group: name, feature, id };
    }
    return null;
  }

  /** Draw a feature over itself in a bright colour, with its edges; null clears. */
  select(hit: Hit | null): void {
    if (this.highlight) {
      this.scene.remove(this.highlight);
      this.highlight.traverse((o) => {
        if (o instanceof Mesh || o instanceof LineSegments) {
          o.geometry.dispose();
          (o.material as MeshBasicMaterial).dispose();
        }
      });
      this.highlight = null;
    }
    const g = hit ? this.groups.get(hit.group) : undefined;
    if (hit && g) {
      const geometry = new BufferGeometry();
      const xyz = g.mesh.geometry.getAttribute("position").array as ArrayLike<number>;
      geometry.setAttribute(
        "position",
        new BufferAttribute(g.index.positions(hit.feature, xyz), 3),
      );
      const fill = new Mesh(
        geometry,
        new MeshBasicMaterial({
          color: HIGHLIGHT,
          transparent: true,
          opacity: 0.45,
          depthWrite: false,
          polygonOffset: true,
          polygonOffsetFactor: -4,
          polygonOffsetUnits: -4,
        }),
      );
      const edges = new LineSegments(
        new EdgesGeometry(geometry, 30),
        new LineBasicMaterial({ color: HIGHLIGHT }),
      );
      this.highlight = new Group().add(fill, edges);
      this.highlight.userData["group"] = hit.group;
      this.highlight.applyMatrix4(g.mesh.matrixWorld);
      this.scene.add(this.highlight);
    }
    this.requestRender();
  }

  /** Where a feature's top (a building's roof) appears on screen, in client pixels. */
  screenPositionOf(id: string): { x: number; y: number } | null {
    for (const g of this.groups.values()) {
      const feature = g.ids.indexOf(id);
      if (feature < 0) continue;
      const xyz = g.mesh.geometry.getAttribute("position").array as ArrayLike<number>;
      const top = g.index.topCentre(feature, xyz);
      if (!top) return null;
      const p = new Vector3(...top).applyMatrix4(g.mesh.matrixWorld).project(this.camera);
      const rect = this.canvas.getBoundingClientRect();
      return {
        x: rect.left + ((p.x + 1) / 2) * rect.width,
        y: rect.top + ((1 - p.y) / 2) * rect.height,
      };
    }
    return null;
  }

  /** Screen angle of north, in degrees clockwise from straight up, for the compass. */
  northAngle(): number {
    const a = this.controls.target.clone().project(this.camera);
    const b = this.controls.target
      .clone()
      .add(new Vector3(0, 0, -10))
      .project(this.camera);
    // Screen coordinates from -1 to 1 stretch differently across and up; undo the stretch.
    return MathUtils.radToDeg(Math.atan2((b.x - a.x) * this.camera.aspect, b.y - a.y));
  }

  requestRender(): void {
    if (this.frameRequested) return;
    this.frameRequested = true;
    requestAnimationFrame(() => {
      this.frameRequested = false;
      const moving = this.controls.update(); // true while the camera is still gliding
      this.renderer.render(this.scene, this.camera);
      this.onRender?.();
      if (moving) this.requestRender();
    });
  }

  /** Width in pixels of a panel over the left of the view; the view centres on the rest. */
  setLeftInset(px: number): void {
    this.leftInset = Math.max(0, px);
    this.resize();
  }

  private resize(): void {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // Shift the picture right by half the panel, so the middle of the uncovered part is the
    // middle of the view. Picking and projecting use the same shifted camera.
    if (this.leftInset > 0) this.camera.setViewOffset(w, h, -this.leftInset / 2, 0, w, h);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }
}
