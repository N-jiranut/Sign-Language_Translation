---
marp: true
theme: default
paginate: true
header: "KMUTT CPE102 | Thai Sign Language Translation (TSL Assist)"
footer: "Progress Presentation Day | Department of Computer Engineering"
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding: 40px 60px;
    background: #0f172a;
    color: #f8fafc;
  }
  h1 {
    color: #38bdf8;
    font-size: 2.2rem;
    margin-bottom: 0.3em;
  }
  h2 {
    color: #f97316;
    font-size: 1.8rem;
    margin-top: 0;
  }
  h3 {
    color: #94a3b8;
    font-size: 1.2rem;
  }
  p, li {
    font-size: 1.15rem;
    line-height: 1.5;
    color: #cbd5e1;
  }
  strong {
    color: #38bdf8;
  }
  .highlight {
    color: #f59e0b;
    font-weight: bold;
  }
  .badge {
    background: #1e293b;
    border: 1px solid #334155;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.9rem;
    color: #38bdf8;
    display: inline-block;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }
  .grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
  }
  .card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
  }
  .metric {
    font-size: 2.4rem;
    font-weight: bold;
    color: #10b981;
    margin: 4px 0;
  }
---

<!-- Slide 1: Title & Core Goal (Criterion 1: Topic Clear 25%) -->
# TSL Assist: Real-Time Thai Sign Language Translation
### ระบบแปลภาษามือไทยแบบเรียลไทม์ผ่านกล้องเพื่อเคาน์เตอร์บริการ

> **"A camera-based assistive system that translates isolated Thai Sign Language words into real-time text and speech for service counters."**
> *(ระบบกล้องอัจฉริยะที่แปลท่าทางภาษามือไทยเป็นข้อความและเสียงพูดแบบเรียลไทม์เพื่อสนับสนุนงานบริการ)*

<br>

* **Objective:** ขจัดอุปสรรคการสื่อสารระหว่างคนหูหนวกและเจ้าหน้าที่บริการด้วย AI ต้นทุนต่ำ
* **Key Tech:** MediaPipe Tasks (112 Features) + PyTorch GRU / Random Forest + Desktop GUI
* **Team:** Computer Engineering (CPE), KMUTT

---

<!-- Slide 2: Real-World Problem & Hook (Criterion 2: Topic Interesting 25%) -->
## 1. The Real-World Problem
### ทำไมการสื่อสารที่เคาน์เตอร์บริการจึงเป็นเรื่องยากสำหรับคนหูหนวก?

<div class="grid-2">
  <div class="card">
    <h3 style="color: #f87171;">⚠️ ปัญหาหน้างานจริง (The Friction)</h3>
    <ul>
      <li><strong>ภาษามือคือภาษาแรก:</strong> ภาษาไทยเขียนเป็นภาษาที่สอง ไวยากรณ์แตกต่างกัน</li>
      <li><strong>ล่ามภาษามือมีจำกัด:</strong> ไม่สามารถประจำทุกจุดบริการได้ตลอด 24 ชม.</li>
      <li><strong>ความล่าช้าในจุดสำคัญ:</strong> การเขียนกระดาษโต้ตอบที่เคาน์เตอร์ รพ. หรือจุดบริการใช้เวลานานและสร้างความอึดอัดให้ทั้งสองฝ่าย</li>
    </ul>
  </div>
  <div class="card">
    <h3 style="color: #38bdf8;">💡 คำถามวิจัย & เป้าหมาย (The Hook)</h3>
    <p><em>"เราจะสร้างระบบกล้องราคาประหยัด ที่อ่านท่าทางสำคัญ 10-15 คำ แล้วแปลเป็นเสียงพูดให้เจ้าหน้าที่เข้าใจได้ทันที โดยไม่ต้องสวมใส่อุปกรณ์เซนเซอร์ได้หรือไม่?"</em></p>
    <div style="margin-top: 12px; background: #0f172a; padding: 10px; border-radius: 8px;">
      <span class="badge">🎯 Focus</span> มุ่งเน้นคำศัพท์สำคัญในจุดบริการ เช่น โรงพยาบาล และจุดต้อนรับ
    </div>
  </div>
</div>

---

<!-- Slide 3: Scope & Boundaries (Criterion 1: Topic Clear 25%) -->
## 2. Project Scope & Vocabulary
### ขอบเขตของระบบที่กำหนดไว้ชัดเจน

<div class="grid-2">
  <div class="card">
    <h3 style="color: #38bdf8;">✅ สิ่งที่ระบบครอบคลุม (In Scope)</h3>
    <ul>
      <li><strong>ประเภทท่าทาง:</strong> คำเดี่ยวหรือวลีสั้น (Isolated Signs)</li>
      <li><strong>คลังคำศัพท์ปัจจุบัน (10 คำ):</strong>
        <br><span style="color: #fbbf24; font-size: 0.95rem;">Done, Dont_like, How, Me, No, Thank, Tomorrow, Worry, Yesterday, You</span>
      </li>
      <li><strong>ตำแหน่งผู้ใช้:</strong> นั่งหน้ากล้องระยะ 90 ซม., กล้องสูง 161 ซม., ก้ม 20°-30°</li>
      <li><strong>การตอบสนอง:</strong> แสดงข้อความบนจอ + เสียงสังเคราะห์ (TTS)</li>
    </ul>
  </div>
  <div class="card">
    <h3 style="color: #94a3b8;">🚫 นอกขอบเขต (Out of Scope)</h3>
    <ul>
      <li><strong>การแปลประโยคยาวต่อเนื่อง:</strong> Continuous sign language ที่ความเร็วธรรมชาติ</li>
      <li><strong>ไวยากรณ์ขั้นสูง:</strong> การถอดรหัสโครงสร้างประโยคภาษามือซับซ้อน</li>
      <li><strong>การแสดงสีหน้า (Facial Expression):</strong> ใช้เฉพาะท่าทางของมือและตำแหน่งแขน/ลำตัว</li>
    </ul>
  </div>
</div>

---

<!-- Slide 4: Roadmap & Agenda (Criterion 1: Topic Clear 25%) -->
## 3. Presentation Roadmap
### ภาพรวมหัวข้อในการนำเสนอความก้าวหน้า

<div class="grid-3">
  <div class="card">
    <div class="badge">Part 1</div>
    <h3 style="color: #38bdf8; margin-top: 8px;">System Architecture</h3>
    <p>สถาปัตยกรรมระบบ 112 Features และการทำ Sequence Resampling 24 เฟรม</p>
  </div>
  <div class="card">
    <div class="badge">Part 2</div>
    <h3 style="color: #10b981; margin-top: 8px;">Current Progress</h3>
    <p>ความสำเร็จ 10 ท่าทาง, Desktop GUI 2 ฝั่ง และระบบจัดตำแหน่งกล้อง HUD</p>
  </div>
  <div class="card">
    <div class="badge">Part 3</div>
    <h3 style="color: #f59e0b; margin-top: 8px;">Lessons & Next Steps</h3>
    <p>ปัญหาที่พบและการแก้โจทย์, แผนงาน Milestone และการทดสอบกับผู้ใช้จริง</p>
  </div>
</div>

---

<!-- Slide 5: Methodology & Pipeline (Criterion 3: Planning 25%) -->
## 4. Methodology: End-to-End Pipeline
### กระบวนการประมวลผลข้อมูลตั้งแต่ภาพจากกล้องจนถึงการแปลผล

```
[Camera Input] ─────────► [MediaPipe Tasks] ─────────► [Scale Normalization]
640x480 @ 30 FPS         Pose (13 pts) + Hands (42 pts)  Normalized by Shoulder & Wrist
                                                                │
                                                                ▼
[Real-Time Output] ◄───── [Classification] ◄────────── [24-Frame Resample]
Display + Audio Voice     Random Forest / PyTorch GRU   Fixed-length Linear Interpolation
```

* **Zero-Lag Camera Threading:** อ่านเฟรมใน Background Thread เพื่อไม่ให้ระบบตรวจจับกระตุก
* **Automatic State Machine:** มีเส้น Ready Zone ระดับอก ยกมือค้าง 0.35 วิเพื่อเริ่มบันทึกอัตโนมัติ

---

<!-- Slide 6: Tangible Progress (Criterion 2: Topic Interesting 25%) -->
## 5. Tangible Progress: What Works Today
### ความก้าวหน้าของระบบที่ทำงานได้จริงในปัจจุบัน

<div class="grid-3">
  <div class="card">
    <div class="badge">Dataset & Model</div>
    <div class="metric">10 Words</div>
    <p>จำแนกได้แม่นยำทั้ง 10 ท่าทาง ด้วย <strong>Random Forest & PyTorch GRU</strong> จากตัวอย่างท่าละ 10 ครั้ง</p>
  </div>
  <div class="card">
    <div class="badge">Desktop Interface</div>
    <div class="metric">Dual-Pane</div>
    <p>UI สองฝั่ง: ด้านซ้ายแสดงภาพกล้องพร้อม Skeleton, ด้านขวาเป็น Conversation Hub & Studio</p>
  </div>
  <div class="card">
    <div class="badge">User Assistance</div>
    <div class="metric">Auto HUD</div>
    <p>ระบบ <strong>Camera Calibration HUD</strong> ตรวจจับกึ่งกลางลำตัวและระยะกางศอกก่อนเริ่มทำท่า</p>
  </div>
</div>

---

<!-- Slide 7: Technical Secret - Why 10 Samples Work (Criterion 2 & 3) -->
## 6. Technical Innovation: Why It Works With 10 Samples
### เคล็ดลับทางวิศวกรรมที่ทำให้โมเดลแม่นยำแม้ข้อมูลมีขนาดกะทัดรัด

<div class="grid-2">
  <div class="card">
    <h3 style="color: #38bdf8;">1. Scale-Invariant Normalization (112 Dims)</h3>
    <ul>
      <li><strong>Pose (26 ค่า):</strong> เทียบพิกัดสัมพัทธ์กับจุดกึ่งกลางและความกว้างไหล่</li>
      <li><strong>Hands (84 ค่า):</strong> ปรับขนาดพิกัดนิ้วด้วยระยะห่างจากข้อมือถึงโคนนิ้วกลาง</li>
      <li><strong>Flags (2 ค่า):</strong> ตรวจจับสถานะการปรากฏของมือซ้าย/ขวา</li>
      <li><em>ผลลัพธ์:</em> ข้อมูลไม่แกว่งตามระยะใกล้-ไกลของกล้อง</li>
    </ul>
  </div>
  <div class="card">
    <h3 style="color: #10b981;">2. Temporal Resampling (24 Frames)</h3>
    <ul>
      <li>ไม่ว่าจะทำท่าเร็ว (1.0 วินาที) หรือช้า (2.0 วินาที)</li>
      <li>ระบบใช้ <strong>Linear Interpolation</strong> บีบ/ยืดให้เป็น 24 เฟรมคงที่</li>
      <li>ขจัดมิติความเร็วที่เหลื่อมกัน ทำให้โมเดลเรียนรู้เฉพาะ <strong>"รูปทรงและทิศทางการเคลื่อนไหว"</strong> ได้อย่างมีประสิทธิภาพ</li>
    </ul>
  </div>
</div>

---

<!-- Slide 8: Lessons & Problem Solving (Criterion 2: Topic Interesting 25%) -->
## 7. Lessons Learned & Engineering Challenges
### ปัญหาจริงที่พบระหว่างพัฒนาและวิธีแก้ไขเชิงวิศวกรรม

| ความท้าทายที่พบ (Blocker) | ผลกระทบ | วิธีการแก้ไข (Solution) |
|---|---|---|
| **ท่าพนมมือ/มือซ้อนกัน (Wai/Clasp)** | ฝ่ามือบังกัน MediaPipe หลุดการตรวจจับ | ปรับมุมก้มกล้อง 20°-30° และให้คำแนะนำเว้นช่องว่างระหว่างฝ่ามือ 1-2 ซม. |
| **สรีระและระยะนั่งที่ไม่เท่ากัน** | ค่าพิกัดเบี่ยงเบน โมเดลทายผิด | สร้างระบบ **Camera Alignment Guide (HUD)** เช็คกึ่งกลางตัวและระยะกางศอก |
| **จุดเริ่ม-จบของท่าทาง (Trigger)** | ท่าทางปะปนกับจังหวะพักมือ | ออกแบบ **Ready Zone State Machine** ตรวจจับการยกมือผ่านระดับอกอัตโนมัติ |

---

<!-- Slide 9: Completed vs. Remaining Work (Criterion 3: Planning 25%) -->
## 8. Milestone Tracking: Completed vs. Next Phase
### การควบคุมไทม์ไลน์และเป้าหมายความสำเร็จของโครงงาน

<div class="grid-2">
  <div class="card">
    <h3 style="color: #10b981;">✅ Completed Work (ความก้าวหน้าที่เสร็จแล้ว)</h3>
    <ul>
      <li>[x] ออกแบบและสร้าง MediaPipe Tasks Feature Pipeline (112 Dims)</li>
      <li>[x] พัฒนาระบบ Temporal Resampling 24 Frames</li>
      <li>[x] พัฒนาและเปรียบเทียบโมเดล Random Forest และ PyTorch GRU</li>
      <li>[x] เก็บข้อมูลชุดแรก 10 คำศัพท์ และทดสอบความแม่นยำ</li>
      <li>[x] พัฒนา Modern Dual-Pane Desktop GUI (TSL Assist)</li>
      <li>[x] สร้างระบบ Camera Alignment & Diagnostic HUD</li>
    </ul>
  </div>
  <div class="card">
    <h3 style="color: #f59e0b;">⏳ Remaining Work (แผนงานระยะถัดไป)</h3>
    <ul>
      <li>[ ] ขยายคลังคำศัพท์เป็น 15-20 คำ (เน้นหมวดสถานพยาบาล/บริการ)</li>
      <li>[ ] เก็บข้อมูลจากผู้ทดสอบหลายคน (Cross-Signer Validation)</li>
      <li>[ ] เชื่อมต่อปุ่มกดทางกายภาพผ่านไมโครคอนโทรลเลอร์ (ESP32)</li>
      <li>[ ] ทดสอบ Usability Test กับสถานการณ์จำลองหน้าเคาน์เตอร์</li>
    </ul>
  </div>
</div>

---

<!-- Slide 10: Concrete Next Steps & Risk Mitigation (Criterion 3: Planning 25%) -->
## 9. Next Steps & Risk Mitigation
### แผนปฏิบัติการที่ชัดเจนและการบริหารความเสี่ยง

<div class="grid-2">
  <div class="card">
    <h3 style="color: #38bdf8;">🎯 แผนการดำเนินงาน 4 สัปดาห์ถัดไป</h3>
    <ul>
      <li><strong>สัปดาห์ที่ 1:</strong> ขยาย Dataset คำศัพท์เพิ่มอีก 5 คำ และทำ Data Augmentation</li>
      <li><strong>สัปดาห์ที่ 2:</strong> ทดสอบโมเดลกับผู้ทดสอบคนใหม่ (Unseen Signers)</li>
      <li><strong>สัปดาห์ที่ 3:</strong> บูรณาการฮาร์ดแวร์ ESP32 Serial Link สำหรับปุ่ม Trigger ภายนอก</li>
      <li><strong>สัปดาห์ที่ 4:</strong> ปรับปรุงส่วนตอบกลับของเจ้าหน้าที่ (Preset Replies) ให้สมบูรณ์</li>
    </ul>
  </div>
  <div class="card">
    <h3 style="color: #ef4444;">🛡️ การรับมือความเสี่ยง (Risk Mitigation)</h3>
    <ul>
      <li><strong>ความเสี่ยง:</strong> โมเดล Overfit กับผู้ทำท่าคนเดิม
        <br><em>วิธีแก้:</em> ใช้ GroupShuffleSplit ในการทดสอบ และเพิ่ม Gaussian Jitter Augmentation
      </li>
      <li><strong>ความเสี่ยง:</strong> แสงสว่างในสถานที่จริงไม่คงที่
        <br><em>วิธีแก้:</em> เตรียมแถบไฟส่องสว่างเสริม (LED Light) และปรับเกณฑ์ Confidence ของ MediaPipe
      </li>
    </ul>
  </div>
</div>

---

<!-- Slide 11: Summary & Takeaway (Criterion 4: Good Presentation 25%) -->
## 10. Summary & Immediate Milestone
### สรุปความก้าวหน้าและก้าวต่อไปของ TSL Assist

<div class="card" style="margin-bottom: 20px; border-left: 4px solid #38bdf8;">
  <h3 style="color: #38bdf8; margin: 0 0 8px 0;">Key Takeaways:</h3>
  <p style="margin: 0;">
    TSL Assist พิสูจน์แล้วว่าการใช้ <strong>MediaPipe 112 Features ร่วมกับ 24-Frame Temporal Resampling</strong> สามารถจำแนกท่าทางภาษามือ 10 คำศัพท์ได้อย่างแม่นยำ ตอบสนองแบบ Real-time และพร้อมใช้งานผ่าน Desktop GUI ที่รองรับการสื่อสารสองทาง
  </p>
</div>

<div class="grid-2">
  <div class="card">
    <h3 style="color: #10b981;">🚀 Next Immediate Milestone</h3>
    <p>เพิ่มคำศัพท์ทางการแพทย์ (Hospital Context) และประเมินค่า Cross-Signer Accuracy ภายในสิ้นเดือนนี้</p>
  </div>
  <div class="card">
    <h3 style="color: #fbbf24;">🤝 Project Impact</h3>
    <p>มุ่งสู่ระบบผู้ช่วยเคาน์เตอร์บริการต้นทุนต่ำ ที่ใช้งานได้จริงและเข้าถึงได้ง่าย</p>
  </div>
</div>
