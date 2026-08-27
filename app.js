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
let evaluation = { frames: 0, faceFrames: 0, earTotal: 0 };
let logs             = [];
let earHistory       = [];
let voiceSettings    = { enabled: false, volume: 0.8, level: 3 };
let voiceTimer       = null;
let currentVoiceLevel = 0;
const voiceMessages   = {
  1: 'Please stay alert.',
  2: 'Drowsiness detected. Please take a break.',
  3: 'Warning. Pull over safely and rest.'
};
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

function openSettings() {
  if (document.getElementById('settingsModal')) return;
  const modal = document.createElement('div');
  modal.id = 'settingsModal';
  modal.className = 'settings-modal';
  modal.innerHTML = `
    <div class="settings-card" role="dialog" aria-modal="true" aria-labelledby="settingsTitle">
      <h3 id="settingsTitle">Detection Settings</h3>
      <label for="settingsSensitivity">Alert sensitivity</label>
      <select id="settingsSensitivity">
        <option value="1">Low</option>
        <option value="2">Medium</option>
        <option value="3">High</option>
      </select>
      <label for="settingsEar">EAR threshold</label>
      <input id="settingsEar" type="number" min="0.1" max="0.5" step="0.01" value="${sens.ear}">
      <label><input id="settingsVoiceEnabled" type="checkbox" ${voiceSettings.enabled ? 'checked' : ''}> Voice alerts</label>
      <label for="settingsVoiceVolume">Voice volume</label>
      <input id="settingsVoiceVolume" type="range" min="0" max="1" step="0.1" value="${voiceSettings.volume}">
      <label for="settingsVoiceLevel">Voice escalation level</label>
      <select id="settingsVoiceLevel">
        <option value="1">1</option><option value="2">2</option><option value="3">3</option>
      </select>
      <div class="settings-actions">
        <button class="btn btn-secondary" onclick="closeSettings()">Cancel</button>
        <button class="btn btn-primary" onclick="saveSettings()">Save</button>
      </div>
    </div>`;
  document.body.appendChild(modal);
  document.getElementById('settingsSensitivity').value = Object.keys(SENSITIVITY).find(key => SENSITIVITY[key] === sens) || '2';
  document.getElementById('settingsVoiceLevel').value = voiceSettings.level;
}

function recordEvaluationSample(ear, faceDetected) {
  evaluation.frames++;
  if (faceDetected) {
    evaluation.faceFrames++;
    evaluation.earTotal += ear;
  }
  updateEvaluation();
}

function updateEvaluation() {
  const hasSamples = evaluation.frames > 0;
  const coverage = hasSamples ? (evaluation.faceFrames / evaluation.frames) * 100 : 0;
  const averageEar = evaluation.faceFrames ? evaluation.earTotal / evaluation.faceFrames : 0;
  const elapsedMinutes = sessionStart ? Math.max((Date.now() - sessionStart) / 60000, 1 / 60) : 0;
  const alertsPerMinute = elapsedMinutes ? alertCount / elapsedMinutes : 0;
  const score = hasSamples ? Math.round((coverage * 0.7) + Math.min(currentFps / 30, 1) * 30) : 0;

  document.getElementById('faceCoverage').textContent = hasSamples ? `${coverage.toFixed(1)}%` : '--';
  document.getElementById('averageEar').textContent = evaluation.faceFrames ? averageEar.toFixed(3) : '--';
  document.getElementById('alertRate').textContent = hasSamples ? alertsPerMinute.toFixed(1) : '--';
  document.getElementById('framesSampled').textContent = evaluation.frames;
  document.getElementById('evaluationScore').textContent = hasSamples ? `${score}/100` : '--';

  const status = document.getElementById('evaluationStatus');
  if (!hasSamples) status.textContent = 'Start detection to evaluate performance';
  else if (coverage >= 90 && currentFps >= 20) status.textContent = 'Healthy: stable face coverage and frame rate';
  else if (coverage >= 70) status.textContent = 'Fair: improve lighting or camera position';
  else status.textContent = 'Needs attention: face coverage is low';
}

function resetEvaluation() {
  evaluation = { frames: 0, faceFrames: 0, earTotal: 0 };
  updateEvaluation();
}

function updateClosureMetrics(closed) {
  const now = Date.now();
  if (closed && !wasEyeClosed) eyeClosedStart = now;
  if (!closed && wasEyeClosed && eyeClosedStart) {
    const duration = (now - eyeClosedStart) / 1000;
    if (duration >= 0.08 && duration < 1.5) blinkCount++;
    if (duration >= 0.5) slowBlinkCount++;
    document.getElementById('blinkCount').textContent = blinkCount;
    document.getElementById('yawnCount').textContent = slowBlinkCount;
    eyeClosedStart = null;
  }
  wasEyeClosed = closed;
  const currentDuration = closed && eyeClosedStart ? (now - eyeClosedStart) / 1000 : 0;
  document.getElementById('closureDuration').textContent = `${currentDuration.toFixed(1)}s`;
  document.getElementById('closureBar').style.width = Math.min(100, (currentDuration / 1.5) * 100) + '%';
  document.getElementById('closureStatus').textContent = closed ? 'Eyes closed' : 'Eyes open';
}

function startDetection() {
  if (isRunning) return;
  isRunning = true;
  sessionStart = Date.now();
  fpsFrames = 0;
  fpsLastTime = performance.now();
  setStatus('Loading detector...', true);
  document.getElementById('startBtn').classList.add('hidden');
  document.getElementById('stopBtn').classList.remove('hidden');
  sessionTimer = setInterval(updateSessionTime, 1000);
  startRealDetection();
}

function stopDetection() {
  isRunning = false;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  modelsLoaded = false;
  if (animFrameId) cancelAnimationFrame(animFrameId);
  if (sessionTimer) clearInterval(sessionTimer);
  animFrameId = null;
  sessionTimer = null;
  setStatus('Detection stopped', false);
  document.getElementById('startBtn').classList.remove('hidden');
  document.getElementById('stopBtn').classList.add('hidden');
}

function resetStats() {
  stopDetection();
  sessionStart = null;
  blinkCount = 0;
  slowBlinkCount = 0;
  alertCount = 0;
  drowsinessScore = 0;
  earBuffer = [];
  eyeClosedStart = null;
  wasEyeClosed = false;
  updateGauge(0);
  resetEvaluation();
  document.getElementById('blinkCount').textContent = '0';
  document.getElementById('yawnCount').textContent = '0';
  document.getElementById('alertCount').textContent = '0';
  document.getElementById('sessionTime').textContent = '00:00';
  document.getElementById('fpsValue').textContent = '--';
  document.getElementById('faceDetected').textContent = 'No';
  document.getElementById('earValue').textContent = '--';
}

async function startRealDetection() {
  try {
    if (!window.faceapi) throw new Error('Face detection library unavailable');
    document.getElementById('loadingMessage').textContent = 'Loading lightweight face models...';
    const modelLoad = Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL)
    ]);
    const timeout = new Promise((resolve, reject) => {
      setTimeout(() => reject(new Error('Model download timed out')), 15000);
    });
    await Promise.race([modelLoad, timeout]);
    modelsLoaded = true;

    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access is unavailable');
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    const video = document.getElementById('video');
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    hideLoading();
    setStatus('Detection active', true);
    detectRealFrame();
  } catch (error) {
    console.error('Real face detection could not start:', error);
    modelsLoaded = false;
    stopDetection();
    document.getElementById('loadingMessage').textContent = `${error.message}. Check network/camera permissions and try again.`;
    document.getElementById('videoLoading').classList.remove('hidden');
    setStatus('Detection unavailable', false);
  }
}

async function detectRealFrame() {
  if (!isRunning || !modelsLoaded) return;
  const video = document.getElementById('video');
  try {
    const result = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 })).withFaceLandmarks(true);
    if (result) {
      const landmarks = result.landmarks;
      const leftEye = landmarks.getLeftEye();
      const rightEye = landmarks.getRightEye();
      const smoothEAR = (computeEAR(leftEye) + computeEAR(rightEye)) / 2;
      recordEvaluationSample(smoothEAR, true);
      document.getElementById('faceDetected').textContent = 'Yes';
      document.getElementById('earValue').textContent = smoothEAR.toFixed(3);
      document.getElementById('earBar').style.width = Math.min(100, (smoothEAR / 0.45) * 100) + '%';
      updateSparkline(smoothEAR);
      const closed = smoothEAR < sens.ear;
      updateClosureMetrics(closed);
      if (closed) {
        drowsinessScore = Math.min(100, drowsinessScore + 2);
        closedFrames++;
        document.getElementById('earStatus').textContent = 'Eyes closing';
      } else {
        drowsinessScore = Math.max(0, drowsinessScore - 1);
        closedFrames = 0;
        document.getElementById('earStatus').textContent = 'Eyes open';
      }
      drawFaceMesh(landmarks, result.detection.box);
      if (closedFrames >= sens.frames) triggerAlert();
      updateGauge(drowsinessScore);
      applyDrowsyState(drowsinessScore);
    } else {
      recordEvaluationSample(0, false);
      document.getElementById('faceDetected').textContent = 'No';
      document.getElementById('earValue').textContent = '--';
      document.getElementById('earStatus').textContent = 'No face detected';
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  } catch (error) {
    console.warn('Detection frame failed:', error);
  }
  calcFPS();
  animFrameId = requestAnimationFrame(detectRealFrame);
}

function updateSessionTime() {
  const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
  const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const s = String(elapsed % 60).padStart(2, '0');
  document.getElementById('sessionTime').textContent = `${m}:${s}`;
}

// -------------------------
// Logging and persistence
// -------------------------
function logEvent(type, data = {}) {
  try {
    logs.push(Object.assign({ time: new Date().toISOString(), type }, data));
  } catch (e) { /* ignore */ }
}

function downloadLogs() {
  if (!logs.length) { alert('No logs to export'); return; }
  const keys = Object.keys(logs[0]);
  const rows = logs.map(r => keys.map(k => JSON.stringify(r[k] ?? '')).join(','));
  const csv = [keys.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'drowseguard_logs.csv';
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

function clearLogs() {
  if (!confirm('Clear all logs?')) return;
  logs.length = 0;
  alert('Logs cleared');
}

 

function closeSettings() {
  const m = document.getElementById('settingsModal');
  if (m) m.parentElement.remove();
}

function saveSettings() {
  const val = document.getElementById('settingsSensitivity').value;
  const earVal = document.getElementById('settingsEar').value;
  sens = SENSITIVITY[val];
  if (earVal) sens.ear = parseFloat(earVal);
  // voice settings
  const vEnabled = document.getElementById('settingsVoiceEnabled').checked;
  const vVolume = parseFloat(document.getElementById('settingsVoiceVolume').value || 0.8);
  const vLevel  = parseInt(document.getElementById('settingsVoiceLevel').value || 3, 10);
  voiceSettings = { enabled: vEnabled, volume: vVolume, level: vLevel };
  localStorage.setItem('drowseguard_sens', JSON.stringify({ val, ear: earVal }));
  localStorage.setItem('drowseguard_voice', JSON.stringify(voiceSettings));
  closeSettings();
}

function loadSettings() {
  try {
    const s = JSON.parse(localStorage.getItem('drowseguard_sens'));
    if (s) {
      sens = SENSITIVITY[s.val || 2];
      if (s.ear) sens.ear = parseFloat(s.ear);
      document.getElementById('sensitivitySlider').value = s.val || 2;
      document.getElementById('sensitivityVal').textContent = SENSITIVITY[s.val || 2].label;
    }
    const vs = JSON.parse(localStorage.getItem('drowseguard_voice'));
    if (vs) voiceSettings = Object.assign(voiceSettings, vs);
  } catch (e) { /* ignore */ }
}

// -------------------------
// Voice escalation
// -------------------------
function speak(text, opts = {}) {
  if (!window.speechSynthesis) return;
  try {
    const u = new SpeechSynthesisUtterance(text);
    u.volume = opts.volume ?? voiceSettings.volume;
    u.rate = opts.rate || 1.0;
    u.pitch = opts.pitch || 1.0;
    window.speechSynthesis.speak(u);
  } catch (e) { /* ignore */ }
}

function startVoiceEscalation() {
  if (!voiceSettings.enabled) return;
  stopVoiceEscalation();
  currentVoiceLevel = 1;
  speak(voiceMessages[currentVoiceLevel], { volume: voiceSettings.volume });
  voiceTimer = setTimeout(() => escalateVoice(), 4000);
}

function escalateVoice() {
  if (!voiceSettings.enabled) return;
  currentVoiceLevel = Math.min(voiceSettings.level, currentVoiceLevel + 1);
  const msg = voiceMessages[currentVoiceLevel] || voiceMessages[voiceSettings.level];
  // increase urgency via rate/pitch for higher levels
  const opts = { volume: voiceSettings.volume, rate: 1 + (currentVoiceLevel - 1) * 0.12, pitch: 1 + (currentVoiceLevel - 1) * 0.08 };
  speak(msg, opts);
  voiceTimer = setTimeout(() => escalateVoice(), 5000);
}

function stopVoiceEscalation() {
  if (voiceTimer) { clearTimeout(voiceTimer); voiceTimer = null; }
  if (window.speechSynthesis) window.speechSynthesis.cancel();
  currentVoiceLevel = 0;
}

// -------------------------
// EAR sparkline
// -------------------------
function updateSparkline(ear) {
  const c = document.getElementById('earSpark');
  if (!c) return;
  const ctx2 = c.getContext('2d');
  earHistory.push(ear);
  if (earHistory.length > 100) earHistory.shift();
  const w = c.width, h = c.height;
  ctx2.clearRect(0,0,w,h);
  // background
  ctx2.fillStyle = 'rgba(255,255,255,0.02)'; ctx2.fillRect(0,0,w,h);
  // path
  ctx2.beginPath();
  for (let i=0;i<earHistory.length;i++) {
    const x = (i/(earHistory.length-1 || 1))*w;
    const v = earHistory[i];
    const y = h - (v/0.45)*h;
    if (i===0) ctx2.moveTo(x,y); else ctx2.lineTo(x,y);
  }
  ctx2.strokeStyle = 'rgba(0,212,170,0.9)'; ctx2.lineWidth = 2; ctx2.stroke();
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
    recordEvaluationSample(smoothEAR, true);

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

  logEvent('alert', { score: drowsinessScore, closedFrames });

  // Sound alert
  playAlertSound();

  // Show overlay
  document.getElementById('alertOverlay').classList.remove('hidden');
  closedFrames = 0;
  startVoiceEscalation();
}

function dismissAlert() {
  document.getElementById('alertOverlay').classList.add('hidden');
  drowsinessScore = Math.max(0, drowsinessScore - 30);
  stopVoiceEscalation();
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


function calcFPS() {
  fpsFrames++;
  const now = performance.now();
  if (now - fpsLastTime >= 1000) {
    currentFps = fpsFrames;
    fpsFrames  = 0;
    fpsLastTime = now;
    document.getElementById('fpsValue').textContent = currentFps + ' fps';
    updateEvaluation();
  }
}

function updateSensitivity(val) {
  sens = SENSITIVITY[val];
  const labels = { 1: 'Low', 2: 'Medium', 3: 'High' };
  document.getElementById('sensitivityVal').textContent = labels[val];
}

// The current browser experience uses the local demo loop, so no model download is needed.
function initializeApp() {
  loadSettings();
  hideLoading();
  setStatus('Ready', false);
  updateEvaluation();
}

initializeApp();
