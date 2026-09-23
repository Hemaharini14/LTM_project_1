import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { ROTATION_LEGS } from '../../data/flightData';

/**
 * Earth for the trip-planner act.
 *
 * No texture downloads: the land is a procedurally generated dot-matrix built
 * from fbm noise on the sphere, which reads as a premium data-viz globe rather
 * than a low-res JPEG of Earth, and costs one InstancedMesh. Night-side falloff
 * comes from a custom fragment shader on the ocean sphere so the terminator is
 * a real lighting gradient, not a texture.
 */

const hash = (x: number, y: number, z: number) => {
  const s = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453;
  return s - Math.floor(s);
};
function noise3(x: number, y: number, z: number) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const xf = x - xi, yf = y - yi, zf = z - zi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf), w = zf * zf * (3 - 2 * zf);
  const l = (a: number, b: number, t: number) => a + (b - a) * t;
  return l(
    l(l(hash(xi, yi, zi), hash(xi + 1, yi, zi), u), l(hash(xi, yi + 1, zi), hash(xi + 1, yi + 1, zi), u), v),
    l(l(hash(xi, yi, zi + 1), hash(xi + 1, yi, zi + 1), u), l(hash(xi, yi + 1, zi + 1), hash(xi + 1, yi + 1, zi + 1), u), v),
    w,
  );
}
const fbm = (x: number, y: number, z: number) =>
  noise3(x, y, z) * 0.55 + noise3(x * 2.1, y * 2.1, z * 2.1) * 0.28 + noise3(x * 4.3, y * 4.3, z * 4.3) * 0.17;

export const latLonToVec3 = (lat: number, lon: number, r: number) => {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(-r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta));
};

const OCEAN_FRAG = `
  varying vec3 vNormal;
  uniform vec3 uLight;
  void main() {
    float d = dot(normalize(vNormal), normalize(uLight));
    float day = smoothstep(-0.25, 0.55, d);
    vec3 night = vec3(0.012, 0.031, 0.063);
    vec3 dayC  = vec3(0.043, 0.125, 0.227);
    // grazing-angle rim, the atmosphere read
    float rim = pow(1.0 - abs(d), 3.0);
    gl_FragColor = vec4(mix(night, dayC, day) + vec3(0.05, 0.42, 0.55) * rim * 0.5, 1.0);
  }`;
const OCEAN_VERT = `
  varying vec3 vNormal;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

function LandDots({ radius, count }: { radius: number; count: number }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  useMemo(() => {
    if (!ref.current) return;
  }, []);

  const { matrices, colors } = useMemo(() => {
    const m: THREE.Matrix4[] = [];
    const c: THREE.Color[] = [];
    const dummy = new THREE.Object3D();
    const golden = Math.PI * (3 - Math.sqrt(5));           // fibonacci sphere = even coverage
    for (let i = 0; i < count; i++) {
      const y = 1 - (i / (count - 1)) * 2;
      const r = Math.sqrt(Math.max(0, 1 - y * y));
      const th = golden * i;
      const p = new THREE.Vector3(Math.cos(th) * r, y, Math.sin(th) * r);
      // fbm threshold carves continents out of the sphere
      if (fbm(p.x * 1.9 + 3, p.y * 1.9, p.z * 1.9) < 0.52) continue;
      dummy.position.copy(p).multiplyScalar(radius);
      dummy.lookAt(0, 0, 0);
      dummy.updateMatrix();
      m.push(dummy.matrix.clone());
      c.push(new THREE.Color().setHSL(0.52, 0.55, 0.32 + Math.abs(y) * 0.18));
    }
    return { matrices: m, colors: c };
  }, [radius, count]);

  return (
    <instancedMesh
      ref={(node) => {
        if (!node) return;
        matrices.forEach((mat, i) => { node.setMatrixAt(i, mat); node.setColorAt(i, colors[i]); });
        node.instanceMatrix.needsUpdate = true;
        if (node.instanceColor) node.instanceColor.needsUpdate = true;
      }}
      args={[undefined, undefined, matrices.length]}
    >
      <circleGeometry args={[0.035, 6]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

/** Great-circle arc between two airports, lifted off the surface. */
function RouteArc({ from, to, radius, delay }: { from: THREE.Vector3; to: THREE.Vector3; radius: number; delay: number }) {
  const ref = useRef<THREE.Line>(null);
  const geo = useMemo(() => {
    const mid = from.clone().add(to).multiplyScalar(0.5).normalize()
      .multiplyScalar(radius + from.distanceTo(to) * 0.42);
    const curve = new THREE.QuadraticBezierCurve3(from, mid, to);
    return new THREE.BufferGeometry().setFromPoints(curve.getPoints(64));
  }, [from, to, radius]);

  useFrame(({ clock }) => {
    if (!ref.current) return;
    const mat = ref.current.material as THREE.LineBasicMaterial;
    // staggered travelling pulse so routes light up in sequence, not together
    const t = (clock.elapsedTime * 0.5 + delay) % 1;
    mat.opacity = 0.18 + Math.sin(t * Math.PI) * 0.6;
  });

  return (
    // @ts-expect-error - three's Line vs SVG line typing collision in R3F
    <line ref={ref} geometry={geo}>
      <lineBasicMaterial color="#22d3ee" transparent opacity={0.4} toneMapped={false} />
    </line>
  );
}

export function Globe({ radius = 9, quality = 1 }: { radius?: number; quality?: number }) {
  const group = useRef<THREE.Group>(null);
  useFrame((_, dt) => { if (group.current) group.current.rotation.y += dt * 0.045; });

  const nodes = useMemo(
    () => ROTATION_LEGS.map((l) => ({ ...l, v: latLonToVec3(l.lat, l.lon, radius * 1.008) })),
    [radius],
  );
  const uniforms = useMemo(() => ({ uLight: { value: new THREE.Vector3(5, 3, 5).normalize() } }), []);

  return (
    <group ref={group}>
      <mesh>
        <sphereGeometry args={[radius, 64, 64]} />
        <shaderMaterial vertexShader={OCEAN_VERT} fragmentShader={OCEAN_FRAG} uniforms={uniforms} />
      </mesh>

      <LandDots radius={radius * 1.002} count={Math.round(9000 * quality)} />

      {/* atmosphere shell */}
      <mesh scale={1.035}>
        <sphereGeometry args={[radius, 48, 48]} />
        <meshBasicMaterial color="#22d3ee" transparent opacity={0.06} side={THREE.BackSide} />
      </mesh>

      {nodes.map((n, i) => (
        <group key={n.code} position={n.v}>
          <mesh><sphereGeometry args={[0.1, 12, 12]} /><meshBasicMaterial color="#67e8f9" toneMapped={false} /></mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.18, 0.24, 24]} />
            <meshBasicMaterial color="#22d3ee" transparent opacity={0.5} side={THREE.DoubleSide} toneMapped={false} />
          </mesh>
          {i < nodes.length - 1 && (
            <RouteArc from={new THREE.Vector3()} to={nodes[i + 1].v.clone().sub(n.v)} radius={radius} delay={i * 0.2} />
          )}
        </group>
      ))}
    </group>
  );
}
