import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * Animated route between two points, with a packet of light travelling along it.
 * The trail is a real curve sampled once; only the packet moves per frame.
 */
export function FlightPath({ from, to, lift = 3, delay = 0, active = false }: {
  from: [number, number, number]; to: [number, number, number]; lift?: number; delay?: number; active?: boolean;
}) {
  const packet = useRef<THREE.Mesh>(null);
  const curve = useMemo(() => {
    const a = new THREE.Vector3(...from), b = new THREE.Vector3(...to);
    const mid = a.clone().add(b).multiplyScalar(0.5).setY(Math.max(a.y, b.y) + lift);
    return new THREE.QuadraticBezierCurve3(a, mid, b);
  }, [from, to, lift]);

  const geo = useMemo(() => new THREE.BufferGeometry().setFromPoints(curve.getPoints(60)), [curve]);
  const tmp = useMemo(() => new THREE.Vector3(), []);

  useFrame(({ clock }) => {
    if (!packet.current) return;
    const t = (clock.elapsedTime * 0.28 + delay) % 1;
    curve.getPoint(t, tmp);
    packet.current.position.copy(tmp);
    (packet.current.material as THREE.MeshBasicMaterial).opacity = Math.sin(t * Math.PI) * 0.95;
  });

  return (
    <group>
      {/* @ts-expect-error three Line vs intrinsic svg line */}
      <line geometry={geo}>
        <lineBasicMaterial color={active ? '#67e8f9' : '#22d3ee'} transparent opacity={active ? 0.7 : 0.28} toneMapped={false} />
      </line>
      <mesh ref={packet}>
        <sphereGeometry args={[0.16, 10, 10]} />
        <meshBasicMaterial color="#a5f3fc" transparent toneMapped={false} />
      </mesh>
    </group>
  );
}
