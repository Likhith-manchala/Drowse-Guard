/* =====================================================
   DrowseGuard – Driver Drowsiness Detection Engine
   ===================================================== */

// ── Constants ─────────────────────────────────────────
const EAR_THRESHOLD       = 0.25;   // below this = eyes closing
const CONSEC_FRAMES_ALERT = 15;     // frames of low EAR before alert
const MODEL_URL           = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';

// Sensitivity presets: [frame threshold multiplier, EAR threshold]
const SENSITIVITY = {
  1: { frames: 25, ear: 0.22, label: 'Low' },
  2: { frames: 15, ear: 0.25, label: 'Medium' },
  3: { frames: 8,  ear: 0.28, label: 'High' },
};

// ── State ─────────────────────────────────────────────
let isRunning        = false;
let animFrameId      = null;
let stream           = null;
let sessionStart     = null;
let sessionTimer     = null;
let earBuffer        = [];          // rolling buffer for smoothing
let closedFrames     = 0;
let blinkCount       = 0;
let slowBlinkCount   = 0;
let alertCount       = 0;
let eyeClosedStart   = null;
let wasEyeClosed     = false;
let drowsinessScore  = 0;           // 0–100
let lastAlertTime    = 0;
let fpsFrames        = 0;
let fpsLastTime      = performance.now();
let currentFps       = 0;
let sens             = SENSITIVITY[2];
let modelsLoaded     = false;

// DOM refs
const video       = document.getElementById('video');
const canvas      = document.getElementById('overlay');
const ctx         = canvas.getContext('2d');

// ── Model Loading ─────────────────────────────────────
window.addEventListener('DOMContentLoaded', loadModels);

async function loadModels() {
  try {
    setStatus('Loading AI models...', false);
    // Wait for face-api to be available (loaded via defer)
    await waitForFaceApi();

    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL);

    modelsLoaded = true;
    setStatus('Ready – Click Start', false);
    hideLoading();
    document.getElementById('startBtn').disabled = false;
  } catch (err) {
    console.error('Model load error:', err);
    // Fallback: use simulated mode for demo
    modelsLoaded = false;
    setStatus('Demo Mode (No Camera)', false);
    hideLoading();
  }
}

function waitForFaceApi() {
  return new Promise((resolve) => {
    const check = () => {
      if (window.faceapi) resolve();
      else setTimeout(check, 100);
    };
    check();
  });
}

// ── Start / Stop ──────────────────────────────────────
async function startDetection() {
  if (isRunning) return;

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise(r => video.onloadedmetadata = r);
    video.play();
  } catch (err) {
    alert('Camera access denied. Please allow camera permissions.');
    return;
  }

  isRunning = true;
  sessionStart = Date.now();
  sessionTimer = setInterval(updateSessionTime, 1000);

  document.getElementById('startBtn').classList.add('hidden');
  document.getElementById('stopBtn').classList.remove('hidden');
  setStatus('Active – Monitoring', true);

  if (modelsLoaded) {
    runDetectionLoop();
  } else {
    runSimulatedLoop();
  }
}

function stopDetection() {
  isRunning = false;
  if (animFrameId) cancelAnimationFrame(animFrameId);
  if (sessionTimer) clearInterval(sessionTimer);
  if (stream) stream.getTracks().forEach(t => t.stop());

  document.getElementById('startBtn').classList.remove('hidden');
  document.getElementById('stopBtn').classList.add('hidden');
  setStatus('Stopped', false);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.body.classList.remove('drowsy-warn', 'drowsy-danger');
  document.getElementById('faceDetected').textContent = 'No';
  document.getElementById('statusDot').className = 'pulse-dot';
}

function resetStats() {
  blinkCount = 0; slowBlinkCount = 0; alertCount = 0;
  closedFrames = 0; drowsinessScore = 0; earBuffer = [];
  updateStatCards(0, 0, '—', '—');
  updateGauge(0);
  document.getElementById('blinkCount').textContent = '0';
  document.getElementById('yawnCount').textContent  = '0';
  document.getElementById('alertCount').textContent = '0';
}

// ── Detection Loop ────────────────────────────────────
async function runDetectionLoop() {
  if (!isRunning) return;

  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  try {
    const det = await faceapi
      .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 320 }))
      .withFaceLandmarks(true);

    if (det) {
      document.getElementById('faceDetected').textContent = 'Yes ✓';
      processLandmarks(det.landmarks);
      drawFaceMesh(det.landmarks, det.detection.box);
    } else {
      document.getElementById('faceDetected').textContent = 'No';
      decayDrowsiness();
    }
  } catch(e) { /* skip frame */ }

  calcFPS();
  animFrameId = requestAnimationFrame(runDetectionLoop);
}

// ── Landmark Processing ───────────────────────────────
function processLandmarks(landmarks) {
  const leftEye  = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();

  const earLeft  = computeEAR(leftEye);
  const earRight = computeEAR(rightEye);
  const ear      = (earLeft + earRight) / 2;

  // Smooth EAR
  earBuffer.push(ear);
  if (earBuffer.length > 8) earBuffer.shift();
  const smoothEAR = earBuffer.reduce((a,b) => a+b) / earBuffer.length;

  // Update EAR display
  document.getElementById('earValue').textContent = smoothEAR.toFixed(3);
  const earPct = Math.min(100, (smoothEAR / 0.45) * 100);
  document.getElementById('earBar').style.width = earPct + '%';

  const eyesClosed = smoothEAR < sens.ear;

  if (eyesClosed) {
    closedFrames++;
    if (!wasEyeClosed) { eyeClosedStart = Date.now(); }
    wasEyeClosed = true;

    const dur = (Date.now() - eyeClosedStart) / 1000;
    document.getElementById('closureDuration').textContent = dur.toFixed(1) + 's';
    const clPct = Math.min(100, (dur / 3) * 100);
    document.getElementById('closureBar').style.width = clPct + '%';
    document.getElementById('closureStatus').textContent = dur > 1 ? '⚠️ Extended closure!' : 'Closing...';
    document.getElementById('earStatus').textContent = '⚠️ Eyes closing';

    drowsinessScore = Math.min(100, drowsinessScore + 3);
  } else {
    if (wasEyeClosed) {
      const dur = (Date.now() - eyeClosedStart) / 1000;
      if (dur < 0.4) blinkCount++;
      else slowBlinkCount++;
      document.getElementById('blinkCount').textContent = blinkCount;
      document.getElementById('yawnCount').textContent  = slowBlinkCount;
      document.getElementById('closureDuration').textContent = '0.0s';
      document.getElementById('closureBar').style.width = '0%';
      document.getElementById('closureStatus').textContent = 'Eyes open';
    }
    wasEyeClosed = false; closedFrames = 0;
    document.getElementById('earStatus').textContent = '✓ Eyes open';
    drowsinessScore = Math.max(0, drowsinessScore - 1.5);
  }

  updateGauge(drowsinessScore);
  applyDrowsyState(drowsinessScore);

  if (closedFrames >= sens.frames) {
    triggerAlert();
  }
}

// Eye Aspect Ratio calculation
function computeEAR(eye) {
  const A = dist(eye[1], eye[5]);
  const B = dist(eye[2], eye[4]);
  const C = dist(eye[0], eye[3]);
  return (A + B) / (2.0 * C);
}
function dist(p1, p2) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

// ── Draw Face Mesh ────────────────────────────────────
function drawFaceMesh(landmarks, box) {
  const pts = landmarks.positions;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Face box glow
  const isDrowsy = drowsinessScore > 50;
  ctx.strokeStyle = isDrowsy ? 'rgba(239,68,68,0.6)' : 'rgba(0,212,170,0.5)';
  ctx.lineWidth = 2;
  ctx.roundRect(box.x, box.y, box.width, box.height, 8);
  ctx.stroke();

  // Draw landmark dots
  ctx.fillStyle = isDrowsy ? 'rgba(239,68,68,0.8)' : 'rgba(0,212,170,0.8)';
  pts.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 1.5, 0, 2 * Math.PI);
    ctx.fill();
  });

  // Draw eye contours
  drawContour(landmarks.getLeftEye(),  isDrowsy);
  drawContour(landmarks.getRightEye(), isDrowsy);
}

function drawContour(points, danger) {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  points.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.closePath();
  ctx.strokeStyle = danger ? 'rgba(239,68,68,0.9)' : 'rgba(0,212,170,0.9)';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.fillStyle = danger ? 'rgba(239,68,68,0.08)' : 'rgba(0,212,170,0.08)';
  ctx.fill();
}

// ── Simulated Demo Mode ───────────────────────────────
function runSimulatedLoop() {
  let t = 0;
  function step() {
    if (!isRunning) return;
    t += 0.02;
    // Simulate a sin-wave EAR that occasionally dips
    const ear = 0.30 + 0.12 * Math.sin(t) - (Math.sin(t * 0.3) > 0.85 ? 0.18 : 0);
    earBuffer.push(ear);
    if (earBuffer.length > 8) earBuffer.shift();
    const smoothEAR = earBuffer.reduce((a,b) => a+b) / earBuffer.length;

    document.getElementById('earValue').textContent = smoothEAR.toFixed(3);
    const earPct = Math.min(100, (smoothEAR / 0.45) * 100);
    document.getElementById('earBar').style.width = earPct + '%';
    document.getElementById('faceDetected').textContent = 'Sim ✓';

    const closed = smoothEAR < sens.ear;
    if (closed) {
      drowsinessScore = Math.min(100, drowsinessScore + 2);
      closedFrames++;
      document.getElementById('earStatus').textContent = '⚠️ Eyes closing (demo)';
    } else {
      drowsinessScore = Math.max(0, drowsinessScore - 1);
      closedFrames = 0;
      document.getElementById('earStatus').textContent = '✓ Eyes open (demo)';
    }
    if (closedFrames >= sens.frames) triggerAlert();

    updateGauge(drowsinessScore);
    applyDrowsyState(drowsinessScore);
    calcFPS();
    animFrameId = requestAnimationFrame(step);
  }
  step();
}

// ── UI Helpers ────────────────────────────────────────
function updateGauge(score) {
  const arc   = document.getElementById('gaugeArc');
  const total = 251;
  const offset = total - (score / 100) * total;
  arc.style.strokeDashoffset = offset;

  document.getElementById('gaugePercent').textContent = Math.round(score);
  const lbl = document.getElementById('gaugeLabel');
  if (score < 30)      { lbl.textContent = 'Alert';    lbl.style.color = '#00d4aa'; }
  else if (score < 60) { lbl.textContent = 'Drowsy';   lbl.style.color = '#f59e0b'; }
  else                 { lbl.textContent = 'Danger!';  lbl.style.color = '#ef4444'; }
}

function applyDrowsyState(score) {
  document.body.classList.remove('drowsy-warn', 'drowsy-danger');
  if (score >= 60) document.body.classList.add('drowsy-danger');
  else if (score >= 30) document.body.classList.add('drowsy-warn');

  const dot = document.getElementById('statusDot');
  if (score >= 60)      { dot.className = 'pulse-dot danger'; }
  else if (score >= 30) { dot.className = 'pulse-dot'; dot.style.background = '#f59e0b'; }
  else if (isRunning)   { dot.className = 'pulse-dot active'; }
}

function decayDrowsiness() {
  drowsinessScore = Math.max(0, drowsinessScore - 0.5);
  updateGauge(drowsinessScore);
}

function triggerAlert() {
  const now = Date.now();
  if (now - lastAlertTime < 8000) return;   // throttle alerts
  lastAlertTime = now;
  alertCount++;
  document.getElementById('alertCount').textContent = alertCount;

  // Sound alert
  playAlertSound();

  // Show overlay
  document.getElementById('alertOverlay').classList.remove('hidden');
  closedFrames = 0;
}

function dismissAlert() {
  document.getElementById('alertOverlay').classList.add('hidden');
  drowsinessScore = Math.max(0, drowsinessScore - 30);
}

function playAlertSound() {
  try {
    const ctx2 = new (window.AudioContext || window.webkitAudioContext)();
    const osc  = ctx2.createOscillator();
    const gain = ctx2.createGain();
    osc.connect(gain); gain.connect(ctx2.destination);
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(440, ctx2.currentTime);
    osc.frequency.linearRampToValueAtTime(880, ctx2.currentTime + 0.3);
    gain.gain.setValueAtTime(0.4, ctx2.currentTime);
    gain.gain.linearRampToValueAtTime(0, ctx2.currentTime + 0.8);
    osc.start(ctx2.currentTime);
    osc.stop(ctx2.currentTime + 0.8);
  } catch(e) {}
}

function updateStatCards(ear, closure, earSt, closeSt) {
  if (earSt)   document.getElementById('earStatus').textContent    = earSt;
  if (closeSt) document.getElementById('closureStatus').textContent = closeSt;
}

function setStatus(msg, active) {
  document.getElementById('statusText').textContent = msg;
  const dot = document.getElementById('statusDot');
  dot.className = active ? 'pulse-dot active' : 'pulse-dot';
  if (!active) dot.style.background = '';
}

function hideLoading() {
  document.getElementById('videoLoading').classList.add('hidden');
}

function updateSessionTime() {
  const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
  const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const s = String(elapsed % 60).padStart(2, '0');
  document.getElementById('sessionTime').textContent = `${m}:${s}`;
}

function calcFPS() {
  fpsFrames++;
  const now = performance.now();
  if (now - fpsLastTime >= 1000) {
    currentFps = fpsFrames;
    fpsFrames  = 0;
    fpsLastTime = now;
    document.getElementById('fpsValue').textContent = currentFps + ' fps';
  }
}

function updateSensitivity(val) {
  sens = SENSITIVITY[val];
  const labels = { 1: 'Low', 2: 'Medium', 3: 'High' };
  document.getElementById('sensitivityVal').textContent = labels[val];
}
