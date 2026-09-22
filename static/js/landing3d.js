/*
 * 3D landing scene for the SmartRouteAI landing page.
 *
 * Renders a real WebGL runway with a continuous stream of airliners on final
 * approach - each one touches down, rolls out past the camera, and the next is
 * already on the glide slope behind it. They're the same trajectory at staggered
 * phases; because z increases monotonically across the whole loop, planes can
 * never overlap and always stay properly separated in trail.
 *
 * The renderer is transparent so the CSS sunset gradient behind it (see .sky in
 * style.css) still shows through as the sky, and the scene fog is matched to that
 * gradient's horizon colour so the far end of the runway dissolves into it.
 *
 * If WebGL is unavailable or three.js failed to load, this does nothing and the
 * CSS/SVG fallback scene stays visible - same "always works, richer if it can"
 * pattern used elsewhere in this app.
 */
(function () {
  "use strict";

  var hero = document.querySelector(".landing-hero");
  var canvas = document.getElementById("landing-canvas");
  if (!hero || !canvas || typeof THREE === "undefined") return;

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Brand palette (mirrors the CSS custom properties in style.css)
  var COLORS = {
    fog: 0xf6a274,
    asphalt: 0x2b3234,
    ground: 0x24403f,
    body: 0xf7f4ef,
    teal: 0x12909e,
    tealDeep: 0x0f6674,
    coral: 0xff7e5f,
    sunset: 0xfeb47b
  };

  var RUNWAY_LENGTH = 900;
  var RUNWAY_WIDTH = 46;
  var TOUCHDOWN_Z = -140;   // where the wheels meet the tarmac
  // Past the camera (z=150) on purpose: the plane exits frame under its own
  // rollout, so the loop restarts off-screen instead of visibly teleporting.
  // Coming back it starts beyond the fog far-plane, so it fades in out of the haze.
  var ROLLOUT_END_Z = 210;
  var START_Z = -900;
  var START_Y = 150;
  var GEAR_HEIGHT = 2.6;    // fuselage centre height when on the ground

  var LOOP_SECONDS = 18;    // one aircraft's full approach -> rollout cycle

  // Each entry is one aircraft in the arrival stream: its phase offset within the
  // loop (evenly spaced = steady arrival interval of LOOP_SECONDS / count) and its
  // livery, so consecutive landings are visibly different aircraft, not a replay.
  var FLEET = [
    { phase: 0.00, stripe: COLORS.coral, tail: COLORS.teal },
    { phase: 0.34, stripe: COLORS.teal, tail: COLORS.coral },
    { phase: 0.67, stripe: COLORS.sunset, tail: COLORS.tealDeep }
  ];

  var renderer, scene, camera, clock;
  var fleet = [];           // { group, phase }

  function makeRunwayTexture() {
    var c = document.createElement("canvas");
    c.width = 256;
    c.height = 4096;
    var g = c.getContext("2d");

    g.fillStyle = "#2b3234";
    g.fillRect(0, 0, c.width, c.height);

    // subtle wear streaks so the asphalt isn't a flat colour
    for (var i = 0; i < 90; i++) {
      g.fillStyle = "rgba(255,255,255," + (0.012 + Math.random() * 0.02).toFixed(3) + ")";
      var w = 6 + Math.random() * 26;
      g.fillRect(Math.random() * c.width, Math.random() * c.height, w, 30 + Math.random() * 260);
    }

    // solid edge lines
    g.fillStyle = "rgba(255,255,255,0.85)";
    g.fillRect(16, 0, 7, c.height);
    g.fillRect(c.width - 23, 0, 7, c.height);

    // dashed centreline
    g.fillStyle = "rgba(255,255,255,0.92)";
    var dash = 150, gap = 130, x = c.width / 2 - 5;
    for (var y = 120; y < c.height - 260; y += dash + gap) {
      g.fillRect(x, y, 10, dash);
    }

    // threshold "piano keys" at both ends
    function keys(topY) {
      g.fillStyle = "rgba(255,255,255,0.9)";
      for (var k = 0; k < 6; k++) {
        g.fillRect(34 + k * 33, topY, 20, 90);
      }
    }
    keys(30);
    keys(c.height - 120);

    var tex = new THREE.CanvasTexture(c);
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    return tex;
  }

  function buildAirliner(livery) {
    var g = new THREE.Group();

    var bodyMat = new THREE.MeshStandardMaterial({ color: COLORS.body, roughness: 0.42, metalness: 0.25 });
    var tealMat = new THREE.MeshStandardMaterial({ color: livery.tail, roughness: 0.45, metalness: 0.2 });
    var coralMat = new THREE.MeshStandardMaterial({ color: livery.stripe, roughness: 0.5, metalness: 0.1 });
    var darkMat = new THREE.MeshStandardMaterial({ color: 0x2f3a3c, roughness: 0.7, metalness: 0.3 });

    // fuselage (cylinders default to the Y axis, so rotate onto Z = direction of travel)
    var fuse = new THREE.Mesh(new THREE.CylinderGeometry(1.7, 1.7, 20, 24), bodyMat);
    fuse.rotation.x = Math.PI / 2;
    g.add(fuse);

    var nose = new THREE.Mesh(new THREE.ConeGeometry(1.7, 4.4, 24), bodyMat);
    nose.rotation.x = Math.PI / 2;
    nose.position.z = 12.2;
    g.add(nose);

    var tailCone = new THREE.Mesh(new THREE.ConeGeometry(1.7, 5.5, 24), bodyMat);
    tailCone.rotation.x = -Math.PI / 2;
    tailCone.position.z = -12.7;
    g.add(tailCone);

    // accent stripe along the fuselage
    var stripe = new THREE.Mesh(new THREE.CylinderGeometry(1.73, 1.73, 20, 24, 1, true,
                                                            Math.PI * 0.86, Math.PI * 0.28), coralMat);
    stripe.rotation.x = Math.PI / 2;
    g.add(stripe);

    // main wings, slightly swept back and dihedral
    function wing(sign) {
      var w = new THREE.Mesh(new THREE.BoxGeometry(17, 0.55, 5.4), bodyMat);
      w.position.set(sign * 9.4, -0.3, -1.4);
      w.rotation.y = sign * -0.19;   // sweep
      w.rotation.z = sign * 0.06;    // dihedral
      w.castShadow = true;
      return w;
    }
    g.add(wing(1));
    g.add(wing(-1));

    // engines slung under the wings
    function engine(sign) {
      var e = new THREE.Mesh(new THREE.CylinderGeometry(1.15, 1.15, 4.6, 18), darkMat);
      e.rotation.x = Math.PI / 2;
      e.position.set(sign * 7.2, -1.7, 0.4);
      e.castShadow = true;
      return e;
    }
    g.add(engine(1));
    g.add(engine(-1));

    // tail surfaces
    var fin = new THREE.Mesh(new THREE.BoxGeometry(0.5, 6, 4.6), tealMat);
    fin.position.set(0, 3.6, -10.4);
    fin.rotation.x = -0.28;
    fin.castShadow = true;
    g.add(fin);

    function stab(sign) {
      var s = new THREE.Mesh(new THREE.BoxGeometry(6.4, 0.42, 2.6), bodyMat);
      s.position.set(sign * 3.6, 0.9, -11);
      s.rotation.y = sign * -0.16;
      s.castShadow = true;
      return s;
    }
    g.add(stab(1));
    g.add(stab(-1));

    fuse.castShadow = true;
    nose.castShadow = true;
    g.scale.setScalar(0.92);
    return g;
  }

  function buildRunwayLights(parent) {
    var lampGeo = new THREE.SphereGeometry(0.5, 8, 8);
    var lampMat = new THREE.MeshBasicMaterial({ color: COLORS.sunset });
    for (var z = -RUNWAY_LENGTH / 2; z < RUNWAY_LENGTH / 2; z += 38) {
      [-1, 1].forEach(function (side) {
        var lamp = new THREE.Mesh(lampGeo, lampMat);
        lamp.position.set(side * (RUNWAY_WIDTH / 2 + 3), 0.6, z);
        parent.add(lamp);
      });
    }
  }

  function init() {
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    scene = new THREE.Scene();
    // matched to the CSS sky gradient's horizon so the far runway melts into it
    scene.fog = new THREE.Fog(COLORS.fog, 320, 900);

    camera = new THREE.PerspectiveCamera(42, 1, 0.1, 2000);
    camera.position.set(62, 20, 150);
    camera.lookAt(0, 6, -90);

    // terrain
    var ground = new THREE.Mesh(
      new THREE.PlaneGeometry(3000, 3000),
      new THREE.MeshStandardMaterial({ color: COLORS.ground, roughness: 1 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.05;
    ground.receiveShadow = true;
    scene.add(ground);

    // runway
    var runway = new THREE.Mesh(
      new THREE.PlaneGeometry(RUNWAY_WIDTH, RUNWAY_LENGTH),
      new THREE.MeshStandardMaterial({ map: makeRunwayTexture(), roughness: 0.92 })
    );
    runway.rotation.x = -Math.PI / 2;
    runway.receiveShadow = true;
    scene.add(runway);

    buildRunwayLights(scene);

    // low warm sunset key light + cool sky fill
    var sun = new THREE.DirectionalLight(0xffb27a, 1.45);
    sun.position.set(-120, 90, -140);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    var s = sun.shadow.camera;
    s.left = -160; s.right = 160; s.top = 160; s.bottom = -160; s.near = 1; s.far = 600;
    scene.add(sun);

    scene.add(new THREE.HemisphereLight(0x9fd6e0, 0x2b3a36, 0.75));

    FLEET.forEach(function (livery) {
      var group = buildAirliner(livery);
      scene.add(group);
      fleet.push({ group: group, phase: livery.phase });
    });

    clock = new THREE.Clock();
    resize();
    window.addEventListener("resize", resize);

    hero.classList.add("three-ready");

    if (reduceMotion) {
      // a single static frame: lead aircraft just before touchdown, the rest in trail
      fleet.forEach(function (a) { updatePlane(a.group, (0.55 + a.phase) % 1); });
      renderer.render(scene, camera);
    } else {
      animate();
    }
  }

  function resize() {
    var w = hero.clientWidth;
    var h = hero.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

  /* p: 0 -> far out on approach, 1 -> rolled out past the camera */
  function updatePlane(plane, p) {
    var APPROACH = 0.62;   // share of the loop spent airborne

    if (p < APPROACH) {
      var t = p / APPROACH;
      plane.position.z = START_Z + (TOUCHDOWN_Z - START_Z) * t;
      // (1-t)^1.7 holds a steady glide slope and only flattens into the flare near
      // the threshold - a plain ease-out drops most of the altitude in the first half
      // and then floats just above the tarmac for the rest, which reads wrong.
      plane.position.y = GEAR_HEIGHT + (START_Y - GEAR_HEIGHT) * Math.pow(1 - t, 1.7);
      // stays nose-down on the glide, pitching up only in the last moments
      plane.rotation.x = -0.10 + 0.16 * Math.pow(t, 2.5);
      plane.rotation.z = Math.sin(t * Math.PI * 1.6) * 0.06 * (1 - t);  // gentle bank, levelling out
      plane.rotation.y = Math.sin(t * Math.PI) * 0.015;
    } else {
      var r = (p - APPROACH) / (1 - APPROACH);
      // decelerating rollout
      plane.position.z = TOUCHDOWN_Z + (ROLLOUT_END_Z - TOUCHDOWN_Z) * easeOutCubic(r);
      plane.position.y = GEAR_HEIGHT;
      plane.rotation.x = 0.06 * (1 - easeOutCubic(r));  // nose lowers onto the gear
      plane.rotation.z = 0;
      plane.rotation.y = 0;
    }
  }

  function animate() {
    requestAnimationFrame(animate);
    var base = (clock.getElapsedTime() % LOOP_SECONDS) / LOOP_SECONDS;
    for (var i = 0; i < fleet.length; i++) {
      updatePlane(fleet[i].group, (base + fleet[i].phase) % 1);
    }
    renderer.render(scene, camera);
  }

  try {
    init();
  } catch (err) {
    // leave the CSS/SVG fallback scene in place rather than showing a broken canvas
    console.warn("[landing3d] WebGL scene unavailable, using CSS fallback:", err);
    hero.classList.remove("three-ready");
  }
})();
