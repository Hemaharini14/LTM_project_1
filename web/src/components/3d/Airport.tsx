import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * Airport node for the rotation map: a glowing marker with an outward radar
 * ping. The ping is driven from a phase offset so a row of nodes pulses in
 * sequence, reading as traffic moving down the route rather than blinking.
 */
export function Airport({ position, phase = 0, active = false, scale = 1 }: {
  position: [number, number, number]; phase?: number; active?: boolean; scale?: number;
}) {
  const ring = useRef<THREE.Mesh>(null);
  const core = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    const t = (clock.elapsedTime * 0.55 + phase) % 1;
    if (ring.current) {
      ring.current.scale.setScalar(1 + t * 3.4);
      (ring.current.material as THREE.MeshBasicMaterial).opacity = (1 - t) * (active ? 0.55 : 0.22);
    }
    if (core.current) {
      const s = 1 + Math.sin(clock.elapsedTime * 2 + phase * 6) * 0.12;
      core.current.scale.setScalar(s);
    }
  });

  const tint = active ? '#67e8f9' : '#22d3ee';
  return (
    <group position={position} scale={scale}>
      <mesh ref={core}>
        <sphereGeometry args={[0.32, 16, 16]} />
        <meshBasicMaterial color={tint} toneMapped={false} />
      </mesh>
      <mesh ref={ring} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.45, 0.58, 40]} />
        <meshBasicMaterial color={tint} transparent opacity={0.4} side={THREE.DoubleSide} toneMapped={false} depthWrite={false} />
      </mesh>
      {/* ground beam */}
      <mesh position={[0, 2.2, 0]}>
        <cylinderGeometry args={[0.035, 0.12, 4.4, 8, 1, true]} />
        <meshBasicMaterial color={tint} transparent opacity={active ? 0.35 : 0.14} depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>
    </group>
  );
}
