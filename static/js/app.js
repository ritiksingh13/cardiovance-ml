(() => {
  const API_URL = "/api/predict";
  const $ = s => document.querySelector(s);
  const form = $("#predictForm");

  const NUMS = {
    Age:{min:1,max:120,label:"Age"}, RestingBP:{min:50,max:250,label:"Blood pressure"},
    Cholesterol:{min:80,max:700,label:"Cholesterol"}, MaxHR:{min:50,max:250,label:"Heart rate"},
    Oldpeak:{min:-3,max:7,label:"ST depression"}
  };
  const RADIOS = ["Gender","FastingBS","ChestPainType","ExerciseAngina","RestingECG","ST_Slope"];
  const chol = $("#Cholesterol"), cholUnknown = $("#cholUnknown");

  /* ---------- validation ---------- */
  function mark(el, msg){
    const f = el.closest(".field");
    f.classList.toggle("bad", !!msg);
    f.querySelector(".err").textContent = msg || "";
  }
  function validate(){
    let ok = true;
    for (const [n,r] of Object.entries(NUMS)){
      const el = form.elements[n];
      if (n === "Cholesterol" && cholUnknown.checked){ mark(el); continue; }
      const raw = el.value.trim(), v = Number(raw);
      if (raw === "" || Number.isNaN(v)){ mark(el, `Please enter your ${r.label.toLowerCase()}.`); ok = false; }
      else if (v < r.min || v > r.max){ mark(el, `Enter a value between ${r.min} and ${r.max}.`); ok = false; }
      else mark(el);
    }
    for (const n of RADIOS){
      const g = form.elements[n];
      if (!g.value){ mark(g[0], "Please choose an option."); ok = false; } else mark(g[0]);
    }
    return ok;
  }
  cholUnknown.addEventListener("change", () => {
    chol.disabled = cholUnknown.checked;
    if (cholUnknown.checked){ chol.value = ""; mark(chol); }
    progress(); 
  });

  /* ---------- progress + ECG sync ---------- */
  function progress(){
    let n = 0;
    for (const k of Object.keys(NUMS)) if (form.elements[k].value.trim() !== "" || (k === "Cholesterol" && cholUnknown.checked)) n++;
    for (const k of RADIOS) if (form.elements[k].value) n++;
    $("#progText").textContent = `${n} of 11 answered`;
    $("#progBar").style.width = (n / 11 * 100) + "%";
  }
  form.addEventListener("input", e => {
    const f = e.target.closest(".field");
    if (f){ f.classList.remove("bad"); f.querySelector(".err").textContent = ""; }
    progress(); syncEcg();
  });
  form.addEventListener("change", () => { progress(); syncEcg(); });

  /* ---------- ECG illustration ---------- */
  const cv = $("#ecg"), cx = cv.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const T = {bpm:72, st:0, slope:0}, C = {...T};
  let W, H, buf, head, phase, acc = 0, last = 0;
  const SPEED = 150;
  const sm = (a,b,x) => { x = Math.min(1, Math.max(0, (x-a)/(b-a))); return x*x*(3-2*x); };
  const gs = (p,c,w,a) => a * Math.exp(-((p-c)**2) / (2*w*w));
  function wave(p, s){
    const win = sm(.31,.35,p) * (1 - sm(.46,.52,p));
    const lvl = (-s.st*.12 + s.slope*.16*((p-.4)/.08)) * win;
    return gs(p,.12,.035,.13) + gs(p,.265,.008,-.12) + gs(p,.285,.009,1) + gs(p,.308,.009,-.26) + lvl + gs(p,.58,.06,.3);
  }
  function push(){ phase = (phase + C.bpm/60/SPEED) % 1; buf[head] = wave(phase, C); head = (head+1) % W; }
  function draw(){
    cx.clearRect(0,0,W,H); cx.beginPath();
    const base = H*.62, A = H*.4;
    for (let x = 0; x < W; x++){ const y = base - buf[(head+x)%W]*A; x ? cx.lineTo(x,y) : cx.moveTo(x,y); }
    cx.lineWidth = 2.2; cx.lineJoin = "round"; cx.strokeStyle = "#1B2340"; cx.stroke();
  }
  function resetEcg(){
    const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1;
    W = Math.round(r.width); H = Math.round(r.height);
    cv.width = W*d; cv.height = H*d; cx.setTransform(d,0,0,d,0,0);
    buf = new Float32Array(W); head = 0; phase = 0; Object.assign(C, T);
    for (let i = 0; i < W; i++) push();
    draw();
  }
  function frame(t){
    const dt = Math.min(.05, (t-last)/1000 || 0); last = t;
    const k = Math.min(1, dt*5);
    for (const key in C) C[key] += (T[key]-C[key])*k;
    acc += dt*SPEED; let n = Math.floor(acc); acc -= n;
    while (n--) push();
    draw(); requestAnimationFrame(frame);
  }
  function syncEcg(){
    const hr = Number(form.MaxHR.value), op = form.Oldpeak.value.trim();
    const hrOk = form.MaxHR.value !== "" && hr >= 50 && hr <= 250;
    const opOk = op !== "" && !Number.isNaN(+op);
    T.bpm = hrOk ? Math.min(hr, 200) : 72;
    T.st = opOk ? Math.max(-1, Math.min(4, +op)) : 0;
    T.slope = {Up:1, Flat:0, Down:-1}[form.ST_Slope.value] ?? 0;
    $("#roHr").textContent = hrOk ? hr : 72;
    $("#roSt").textContent = (opOk ? +op : 0).toFixed(1);
    $("#roNote").textContent = hrOk ? "from your inputs" : "preview";
    if (reduce) resetEcg();
  }
  addEventListener("resize", resetEcg);
  resetEcg();
  if (!reduce) requestAnimationFrame(frame);

  /* ---------- submit ---------- */
  function payload(){
    return {
      Age:Number(form.Age.value), Gender:form.Gender.value, ChestPainType:form.ChestPainType.value,
      RestingBP:Number(form.RestingBP.value),
      Cholesterol: cholUnknown.checked ? 0 : Number(chol.value),
      FastingBS:Number(form.FastingBS.value), RestingECG:form.RestingECG.value, MaxHR:Number(form.MaxHR.value),
      ExerciseAngina:form.ExerciseAngina.value, Oldpeak:Number(form.Oldpeak.value), ST_Slope:form.ST_Slope.value
    };
  }
  form.addEventListener("submit", async e => {
    e.preventDefault();
    if (!validate()){ const f = form.querySelector(".field.bad"); f && f.scrollIntoView({behavior:"smooth", block:"center"}); return; }
    const btn = $("#submitBtn"); btn.disabled = true; btn.textContent = "Checking...";
    try {
      const res = await fetch(API_URL, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload())});
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `The server returned an error (${res.status}).`);
      showResult(data);
    } catch (err) { showError(err); }
    finally { btn.disabled = false; btn.textContent = "Check my risk"; }
  });

  function showResult(d){
    const high = Number(d.prediction) === 1;
    const pct = Math.round(d.probability * 100);
    const R = 54, circ = 2*Math.PI*R, box = $("#result");
    box.className = "result " + (high ? "high" : "low");
    box.innerHTML = `
      <div class="gauge"><svg viewBox="0 0 150 150" aria-hidden="true"><circle class="bg" cx="75" cy="75" r="${R}"/>
        <circle class="fg" cx="75" cy="75" r="${R}" stroke-dasharray="${circ}" stroke-dashoffset="${circ}"/></svg><b>${pct}%</b></div>
      <div>
        <h3>${high ? "Higher risk of heart disease" : "Lower risk of heart disease"}</h3>
        <p>${high
          ? "Your answers are closer to those of patients who had heart disease. Please share your reports with a doctor, ideally a cardiologist, for a proper assessment."
          : "Your answers are closer to those of patients who did not have heart disease. Keep up regular check-ups, and see a doctor promptly if you get chest pain, breathlessness or dizziness."}</p>
        ${d.cholesterol_estimated ? '<p class="fine">You did not enter cholesterol, so an average value was used and the estimate is less reliable.</p>' : ""}
        <p class="fine">Estimated probability of heart disease: ${pct}%. This is a statistical estimate from a machine learning model, not a diagnosis.</p>
      </div>`;
    box.hidden = false;
    box.scrollIntoView({behavior:"smooth", block:"center"});
    requestAnimationFrame(() => requestAnimationFrame(() => { box.querySelector(".fg").style.strokeDashoffset = circ * (1 - pct/100); }));
  }
  function showError(err){
    const box = $("#result"); box.className = "result error";
    box.innerHTML = `<h3>We couldn't get a result</h3><p>${err.message} Please check your answers and try again. If the problem continues, wait a minute and retry, as the server may be starting up.</p>`;
    box.hidden = false; box.scrollIntoView({behavior:"smooth", block:"center"});
  }

  $("#resetBtn").addEventListener("click", () => setTimeout(() => {
    chol.disabled = false; $("#result").hidden = true;
    form.querySelectorAll(".field").forEach(f => { f.classList.remove("bad"); f.querySelector(".err").textContent = ""; });
    progress(); syncEcg();
  }, 0));

  progress(); syncEcg();
})();
