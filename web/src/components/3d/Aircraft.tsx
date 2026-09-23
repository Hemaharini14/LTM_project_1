import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * Procedurally built narrow-body airliner.
 *
 * Built from lathed/extruded geometry rather than a downloaded GLB so it loads
 * instantly, has no licensing question, and can be tuned to real proportions -
 * a 737-ish 1:9.5 fuselage fineness ratio, 25-degree wing sweep, 5-degree
 * dihedral, underslung nacelles forward of the wing. Toy aircraft read wrong
 * because they get those ratios wrong, not because they lack polygons.
 *
 * Materials are physically-based and lean on the scene's environment map for
 * reflections; the fuselage is a brushed aluminium (high metalness, low-ish
 * roughness) so the rim light reads as a specular edge rather than a glow.
 */

const SKIN = { color: '#cfd8e3', metalness: 0.92, roughness: 0.28 };
const DARK = { color: '#121a26', metalness: 0.75, roughness: 0.45 };
const ACCENT = { color: '#22d3ee', metalness: 0.4, roughness: 0.3 };

/** Lathed fuselage: nose taper -> constant section -> upswept tail. */
function useFuselageGeometry(detail: number) {
  return useMemo(() => {
    const pts: THREE.Vector2[] = [];
    const L = 20;
    for (let i = 0; i <= 44; i++) {
      const t = i / 44;
      const x = t * L - L / 2;
      let r: number;
      if (t < 0.13) {
        // nose: elliptical taper, not a cone
        r = 1.28 * Math.sqrt(Math.max(0, 1 - Math.pow(1 - t / 0.13, 2))) * 0.98;
      } else if (t < 0.7) {
        r = 1.28;
      } else {
        // tailcone sweeps up and narrows
        const k = (t - 0.7) / 0.3;
        r = 1.28 * (1 - 0.82 * k * k);
      }
      pts.push(new THREE.Vector2(Math.max(r, 0.04), x));
    }
    const g = new THREE.LatheGeometry(pts, detail);
    g.rotateZ(Math.PI / 2);
    return g;
  }, [detail]);
}

/** Tapered, swept wing built as an extruded planform (not a scaled box). */
function useWingGeometry() {
  return useMemo(() => {
    const s = new THREE.Shape();
    s.moveTo(0, 1.9);          // root leading edge
    s.lineTo(8.6, 0.55);       // tip leading edge (sweep)
    s.lineTo(9.0, 0.02);       // tip chord
    s.lineTo(0, -1.5);         // root trailing edge
    s.closePath();
    const g = new THREE.ExtrudeGeometry(s, { depth: 0.17, bevelEnabled: true, bevelSize: 0.05, bevelThickness: 0.05, bevelSegments: 2 });
    g.rotateX(Math.PI / 2);
    return g;
  }, []);
}

function Engine({ side, quality }: { side: 1 | -1; quality: number }) {
  return (
    <group position={[-0.6, -1.05, side * 3.5]} rotation={[0, 0, 0]}>
      {/* nacelle */}
      <mesh castShadow>
        <cylinderGeometry args={[0.82, 0.72, 3.1, quality, 1, true]} />
        <meshStandardMaterial {...SKIN} side={THREE.DoubleSide} />
      </mesh>
      {/* intake lip */}
      <mesh position={[1.55, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <torusGeometry args={[0.8, 0.09, 10, quality]} />
        <meshStandardMaterial {...DARK} />
      </mesh>
      {/* fan face - faintly self-lit so the engine reads as live */}
      <mesh position={[1.5, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <circleGeometry args={[0.74, quality]} />
        <meshStandardMaterial color="#0b1420" emissive="#22d3ee" emissiveIntensity={0.35} metalness={0.9} roughness={0.35} />
      </mesh>
      {/* exhaust glow */}
      <mesh position={[-1.7, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <circleGeometry args={[0.5, quality]} />
        <meshBasicMaterial color="#67e8f9" transparent opacity={0.5} />
      </mesh>
      {/* pylon */}
      <mesh position={[-0.35, 0.72, 0]} castShadow>
        <boxGeometry args={[1.5, 1.0, 0.18]} />
        <meshStandardMaterial {...SKIN} />
      </mesh>
    </group>
  );
}

export function Aircraft({ quality = 24 }: { quality?: number }) {
  const group = useRef<THREE.Group>(null);
  const fuselage = useFuselageGeometry(quality);
  const wing = useWingGeometry();

  // Gentle life: a long-period bank and bob so it never looks frozen, but slow
  // enough that it never competes with the scroll-driven camera.
  useFrame(({ clock }) => {
    if (!group.current) return;
    const t = clock.elapsedTime;
    group.current.rotation.z = Math.sin(t * 0.22) * 0.045;
    group.current.position.y = Math.sin(t * 0.4) * 0.16;
  });

  return (
    <group ref={group} dispose={null}>
      <mesh geometry={fuselage} castShadow receiveShadow>
        <meshStandardMaterial {...SKIN} />
      </mesh>

      {/* cockpit glazing */}
      <mesh position={[8.3, 0.52, 0]} rotation={[0, 0, -0.22]}>
        <sphereGeometry args={[0.66, quality, quality, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#060d18" metalness={1} roughness={0.08} />
      </mesh>

      {/* cheatline */}
      <mesh>
        <cylinderGeometry args={[1.292, 1.292, 13, quality, 1, true, 0, 0.34]} />
        <meshStandardMaterial {...ACCENT} emissive="#22d3ee" emissiveIntensity={0.25} side={THREE.DoubleSide} />
      </mesh>

      {[1, -1].map((side) => (
        <group key={side}>
          {/* wing: swept back, 5deg dihedral */}
          <group position={[-0.8, -0.55, 0]} rotation={[side === 1 ? -0.088 : Math.PI + 0.088, 0, 0]}>
            <mesh geometry={wing} castShadow receiveShadow scale={[1, 1, side]}>
              <meshStandardMaterial {...SKIN} />
            </mesh>
            {/* winglet */}
            <mesh position={[-8.4, 0.35, 0.1]} rotation={[0, 0, 0.32]} castShadow>
              <boxGeometry args={[1.5, 1.35, 0.12]} />
              <meshStandardMaterial {...ACCENT} />
            </mesh>
          </group>
          <Engine side={side as 1 | -1} quality={quality} />
          {/* horizontal stabiliser */}
          <mesh position={[-8.4, 0.35, side * 1.9]} rotation={[0, side * 0.24, 0]} castShadow>
            <boxGeometry args={[2.5, 0.12, 3.6]} />
            <meshStandardMaterial {...SKIN} />
          </mesh>
        </group>
      ))}

      {/* vertical fin */}
      <mesh position={[-8.7, 2.2, 0]} rotation={[0, 0, -0.34]} castShadow>
        <boxGeometry args={[3.5, 4.0, 0.16]} />
        <meshStandardMaterial {...SKIN} />
      </mesh>
      <mesh position={[-9.4, 3.5, 0]} rotation={[0, 0, -0.34]}>
        <boxGeometry args={[1.7, 1.7, 0.2]} />
        <meshStandardMaterial {...ACCENT} emissive="#22d3ee" emissiveIntensity={0.5} />
      </mesh>

      {/* navigation lights */}
      <mesh position={[-1.0, -0.55, 9.1]}><sphereGeometry args={[0.1, 8, 8]} /><meshBasicMaterial color="#ff4d4d" /></mesh>
      <mesh position={[-1.0, -0.55, -9.1]}><sphereGeometry args={[0.1, 8, 8]} /><meshBasicMaterial color="#4dff88" /></mesh>
    </group>
  );
}
