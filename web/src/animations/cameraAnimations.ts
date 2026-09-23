/**
 * Camera choreography.
 *
 * Each beat is a keyframe of position + look-at target; the rig damps between
 * them rather than cutting, so scrubbing backwards is as smooth as forwards and
 * there are never hard scene changes. Hand-placed to read cinematically:
 * three-quarter hero, push in to the forward fuselage, lift overhead for the
 * rotation map, then pull back and off the aircraft entirely for the globe.
 */
import * as THREE from 'three';
import { BEATS, clamp, type SectionId } from './scrollAnimations';

type Key = { at: number; pos: [number, number, number]; look: [number, number, number]; fov: number };

export const CAMERA_KEYS: Key[] = [
  { at: BEATS.hero,        pos: [16, 4.5, 26],   look: [0, 0, 0],      fov: 42 },
  { at: BEATS.prediction,  pos: [10, 1.5, 15],   look: [2, 0, 0],      fov: 38 },
  { at: BEATS.causes,      pos: [-2, 3, 21],     look: [0, 0.5, 0],    fov: 46 },
  { at: BEATS.rotation,    pos: [0, 30, 14],     look: [0, 0, -2],     fov: 50 },
  { at: BEATS.explanation, pos: [7, 2, 17],      look: [0, 0, 0],      fov: 40 },
  { at: BEATS.planner,     pos: [0, 2, 46],      look: [0, -2, 0],     fov: 45 },
  { at: BEATS.trip,        pos: [6, 0, 30],      look: [0, -2, 0],     fov: 42 },
  { at: BEATS.itinerary,   pos: [-8, 3, 34],     look: [0, -2, 0],     fov: 44 },
  { at: BEATS.cta,         pos: [0, 8, 62],      look: [0, -4, 0],     fov: 50 },
];

const _pos = new THREE.Vector3();
const _look = new THREE.Vector3();

/** Interpolated camera state for a scroll position. */
export function cameraAt(p: number) {
  let a = CAMERA_KEYS[0], b = CAMERA_KEYS[0];
  for (let i = 0; i < CAMERA_KEYS.length - 1; i++) {
    if (p >= CAMERA_KEYS[i].at && p <= CAMERA_KEYS[i + 1].at) { a = CAMERA_KEYS[i]; b = CAMERA_KEYS[i + 1]; break; }
    if (p > CAMERA_KEYS[i + 1].at) { a = b = CAMERA_KEYS[i + 1]; }
  }
  const span = b.at - a.at;
  const raw = span > 0 ? clamp((p - a.at) / span) : 0;
  const t = raw * raw * (3 - 2 * raw);           // ease both ends of every move
  _pos.set(lerp3(a.pos, b.pos, t)[0], lerp3(a.pos, b.pos, t)[1], lerp3(a.pos, b.pos, t)[2]);
  _look.set(lerp3(a.look, b.look, t)[0], lerp3(a.look, b.look, t)[1], lerp3(a.look, b.look, t)[2]);
  return { pos: _pos, look: _look, fov: a.fov + (b.fov - a.fov) * t };
}

function lerp3(a: [number, number, number], b: [number, number, number], t: number) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t] as const;
}

/** Aircraft pose per beat - it banks away as the story moves to the globe. */
export function aircraftPose(p: number) {
  const toGlobe = clamp((p - BEATS.planner + 0.08) / 0.16);
  return {
    y: -toGlobe * 26,                       // climbs out of frame for the globe act
    rotY: -0.5 + p * 2.6,
    bank: Math.sin(p * 7) * 0.08 - toGlobe * 0.5,
    visible: toGlobe < 0.98,
  };
}

export const globeReveal = (p: number, s: SectionId = 'planner') =>
  clamp((p - BEATS[s] + 0.1) / 0.18);
