import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * Atmospheric dust + distant airport lights.
 *
 * One Points cloud, no per-frame allocation - the drift is done in the shader-free
 * path by rotating the whole cloud, which is effectively free next to updating
 * thousands of positions on the CPU each frame.
 */
export function Particles({ count = 900, radius = 70 }: { count?: number; radius?: number }) {
  const ref = useRef<THREE.Points>(null);

  const geometry = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const size = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      // biased toward a shell so the camera isn't constantly inside the dust
      const r = radius * (0.35 + Math.random() * 0.65);
      const th = Math.random() * Math.PI * 2;
      const ph = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = r * Math.sin(ph) * Math.cos(th);
      pos[i * 3 + 1] = r * Math.cos(ph) * 0.45;   // flattened: reads as haze layers
      pos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th);
      size[i] = Math.random() * 0.18 + 0.03;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    return g;
  }, [count, radius]);

  useFrame((_, dt) => {
    if (ref.current) { ref.current.rotation.y += dt * 0.012; ref.current.rotation.x += dt * 0.004; }
  });

  return (
    <points ref={ref} geometry={geometry}>
      <pointsMaterial size={0.13} color="#7dd3fc" transparent opacity={0.5} sizeAttenuation depthWrite={false} blending={THREE.AdditiveBlending} />
    </points>
  );
}

/** Slow volumetric-ish haze: a few large, very transparent billboards. */
export function Haze({ count = 7 }: { count?: number }) {
  const group = useRef<THREE.Group>(null);
  const planes = useMemo(
    () => Array.from({ length: count }, (_, i) => ({
      pos: [(Math.random() - 0.5) * 70, -6 - Math.random() * 10, -12 - i * 9] as [number, number, number],
      scale: 26 + Math.random() * 26,
      opacity: 0.035 + Math.random() * 0.045,
    })),
    [count],
  );
  useFrame(({ clock }) => {
    if (!group.current) return;
    group.current.children.forEach((c, i) => { c.position.x += Math.sin(clock.elapsedTime * 0.06 + i) * 0.006; });
  });
  return (
    <group ref={group}>
      {planes.map((p, i) => (
        <mesh key={i} position={p.pos} scale={p.scale}>
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial color="#0e4a6b" transparent opacity={p.opacity} depthWrite={false} blending={THREE.AdditiveBlending} />
        </mesh>
      ))}
    </group>
  );
}
