import { Suspense, useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Environment, AdaptiveDpr, AdaptiveEvents, Preload } from '@react-three/drei';
import * as THREE from 'three';

import { Aircraft } from './3d/Aircraft';
import { Globe, latLonToVec3 } from './3d/Globe';
import { Particles, Haze } from './3d/Particles';
import { Airport } from './3d/Airport';
import { FlightPath } from './3d/FlightPath';
import { cameraAt, applyBias, aircraftPose, globeReveal } from '../animations/cameraAnimations';
import { BEATS, damp, range } from '../animations/scrollAnimations';
import { ROTATION_LEGS } from '../data/flightData';

/**
 * The single persistent 3D scene behind the whole page.
 *
 * Everything is driven from one scroll value: the camera rig damps toward the
 * keyframe for the current position, the aircraft climbs away as the globe act
 * begins, and the rotation map fades in only for its beat. Keeping it to one
 * Canvas for the whole story is what makes the transitions continuous - a
 * canvas per section would mean a hard cut at every boundary.
 *
 * quality is decided once from device capability and scales geometry detail,
 * particle counts and shadow work rather than switching features on and off,
 * so a phone gets the same story at a lower cost.
 */

type Props = { progressRef: React.MutableRefObject<number>; quality: number };

function CameraRig({ progressRef }: { progressRef: React.MutableRefObject<number> }) {
  const { camera, size } = useThree();
  const look = useRef(new THREE.Vector3());

  useFrame((_, dt) => {
    const { pos, look: target, fov, bias } = cameraAt(progressRef.current);
    // Compose the shot: hold the subject clear of the copy on wide screens,
    // recentre it once the layout stacks.
    applyBias(pos, target, bias, size.width / size.height, size.width);
    // Damped, not snapped: scrubbing the scrollbar fast stays smooth and
    // reversing direction doesn't jerk.
    const k = 3.2;
    camera.position.set(
      damp(camera.position.x, pos.x, k, dt),
      damp(camera.position.y, pos.y, k, dt),
      damp(camera.position.z, pos.z, k, dt),
    );
    look.current.set(
      damp(look.current.x, target.x, k, dt),
      damp(look.current.y, target.y, k, dt),
      damp(look.current.z, target.z, k, dt),
    );
    camera.lookAt(look.current);
    const cam = camera as THREE.PerspectiveCamera;
    const nextFov = damp(cam.fov, fov, k, dt);
    if (Math.abs(nextFov - cam.fov) > 0.01) { cam.fov = nextFov; cam.updateProjectionMatrix(); }
  });
  return null;
}

/** Aircraft + its pose, lifted out so the rig can drive it without re-rendering React. */
function Subject({ progressRef, quality }: Props) {
  const group = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    if (!group.current) return;
    const pose = aircraftPose(progressRef.current);
    group.current.position.y = damp(group.current.position.y, pose.y, 2.4, dt);
    group.current.rotation.y = damp(group.current.rotation.y, pose.rotY, 2.0, dt);
    group.current.rotation.z = damp(group.current.rotation.z, pose.bank, 2.0, dt);
    group.current.visible = pose.visible;
  });
  return (
    <group ref={group}>
      <Aircraft quality={quality > 0.6 ? 28 : 14} />
    </group>
  );
}

/** The aircraft-rotation map: only mounted work is the fade, geometry is static. */
function RotationMap({ progressRef }: { progressRef: React.MutableRefObject<number> }) {
  const group = useRef<THREE.Group>(null);

  const nodes = useMemo(() => {
    // Lay the real Indian network out on a plane, scaled from true lat/lon so
    // the shape is geographically honest rather than decorative.
    const v = ROTATION_LEGS.map((l) => latLonToVec3(l.lat, l.lon, 1));
    return ROTATION_LEGS.map((l, i) => ({
      ...l,
      p: [v[i].x * 26, -6, -v[i].z * 26] as [number, number, number],
    }));
  }, []);

  useFrame((_, dt) => {
    if (!group.current) return;
    const p = progressRef.current;
    const vis = range(p, BEATS.causes + 0.06, BEATS.rotation) * (1 - range(p, BEATS.explanation - 0.02, BEATS.explanation + 0.06));
    group.current.visible = vis > 0.01;
    group.current.scale.setScalar(damp(group.current.scale.x || 0.001, 0.3 + vis * 0.7, 3, dt));
  });

  return (
    <group ref={group}>
      {nodes.map((n, i) => (
        <group key={n.code}>
          <Airport position={n.p} phase={i * 0.18} active={i === 0 || i === nodes.length - 1} />
          {i < nodes.length - 1 && (
            <FlightPath from={n.p} to={nodes[i + 1].p} lift={5} delay={i * 0.22} active={i === nodes.length - 2} />
          )}
        </group>
      ))}
    </group>
  );
}

function GlobeAct({ progressRef, quality }: Props) {
  const group = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    if (!group.current) return;
    const r = globeReveal(progressRef.current);
    group.current.visible = r > 0.01;
    const s = 0.4 + r * 0.6;
    group.current.scale.setScalar(damp(group.current.scale.x || 0.001, s, 2.5, dt));
    group.current.position.y = damp(group.current.position.y, -2 - (1 - r) * 12, 2.2, dt);
  });
  return (
    <group ref={group} position={[0, -2, 0]}>
      <Globe radius={9} quality={quality} />
    </group>
  );
}

function Lighting({ quality }: { quality: number }) {
  return (
    <>
      {/* key: warm-neutral, high and forward */}
      <directionalLight
        position={[14, 18, 10]} intensity={2.3} color="#eaf2ff"
        castShadow={quality > 0.6}
        shadow-mapSize={quality > 0.6 ? [1024, 1024] : [512, 512]}
        shadow-camera-far={90}
      />
      {/* cyan rim from behind-left: the aerospace read */}
      <directionalLight position={[-18, 4, -14]} intensity={2.6} color="#22d3ee" />
      {/* cool underfill so the belly never goes black */}
      <directionalLight position={[0, -12, 6]} intensity={0.5} color="#1e3a5f" />
      <ambientLight intensity={0.12} />
    </>
  );
}

export function AircraftScene({ progressRef, quality }: Props) {
  return (
    <div className="fixed inset-0 z-0">
      <Canvas
        shadows={quality > 0.6}
        dpr={[1, quality > 0.6 ? 2 : 1.4]}          // never render above 2x, phones capped lower
        gl={{ antialias: quality > 0.4, powerPreference: 'high-performance', alpha: false }}
        camera={{ position: [16, 4.5, 26], fov: 42, near: 0.1, far: 400 }}
        onCreated={({ gl, scene }) => {
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.05;
          scene.fog = new THREE.FogExp2('#03070d', 0.0115);
        }}
      >
        <color attach="background" args={['#03070d']} />
        <Suspense fallback={null}>
          <Lighting quality={quality} />
          {/* image-based lighting gives the fuselage something real to reflect */}
          <Environment preset="night" />
          <CameraRig progressRef={progressRef} />
          <Subject progressRef={progressRef} quality={quality} />
          <RotationMap progressRef={progressRef} />
          <GlobeAct progressRef={progressRef} quality={quality} />
          <Particles count={Math.round(900 * quality)} />
          {quality > 0.5 && <Haze count={Math.round(7 * quality)} />}
          <Preload all />
        </Suspense>
        <AdaptiveDpr pixelated />
        <AdaptiveEvents />
      </Canvas>
    </div>
  );
}
