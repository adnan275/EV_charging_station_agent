# Sentinel: SOLID Principles & Design Patterns

Ye document explain karta hai ki **Sentinel** project ke architecture mein kaun-kaun se **SOLID Principles** aur **Software Design Patterns** use kiye gaye hain. Inhi principles ki wajah se project highly modular, maintainable, aur safe banta hai.

---

## 🏗️ 1. SOLID Principles Used

Sentinel ka Core Engine (Python) object-oriented design ke best practices ko follow karta hai.

### S - Single Responsibility Principle (SRP)
**Concept:** Har class ka sirf ek hi responsibility (kaam) hona chahiye.
**How Sentinel Uses It:**
- `Scanner`: Iska kaam sirf files dhoondhna aur unka metadata nikalna hai. Ye decision nahi leta ki in files ka kya karna hai.
- `AIPlanner`: Iska kaam sirf JSON plan banana hai. Ye files ko move nahi karta.
- `SafetyValidator`: Ye sirf plan ki safety check karta hai. Ye execution ya planning me involve nahi hota.
- `Executor`: Ye sirf commands (move, rename, delete) run karta hai. Ye planning nahi karta.

### O - Open/Closed Principle (OCP)
**Concept:** Software entities extend karne ke liye open honi chahiye, lekin modify karne ke liye closed.
**How Sentinel Uses It:**
- `RulesEngine`: Agar kal ko naye file types (e.g., 3D models `.obj`) ke liye rules add karne hain, toh hume existing core logic change karne ki zaroorat nahi. Hum sirf YAML ya DB me naye rules inject kar sakte hain.
- **Multiple UIs:** API layer (FastAPI) aisi banayi gayi hai ki kal ko agar Mobile App banani ho, toh core logic modify kiye bina naya UI attach kiya ja sakta hai.

### L - Liskov Substitution Principle (LSP)
**Concept:** Subtypes must be substitutable for their base types.
**How Sentinel Uses It:**
- `PlanSchema`: Pipeline ko is se farq nahi padta ki plan `AIPlanner` ne banaya hai ya (fallback) `RulesEngine` ne. Jab tak output `PlanSchema` model ko match karta hai, `SafetyValidator` aur `Executor` usey smoothly handle kar lenge.

### I - Interface Segregation Principle (ISP)
**Concept:** Clients ko aise interfaces pe depend nahi karna chahiye jo wo use nahi karte.
**How Sentinel Uses It:**
- **API Endpoints:** SentinelAPI me ek bada bloated endpoint banne ke bajaye chhote, specific endpoints hain (`/scan`, `/plan`, `/execute`, `/undo`). CLI client sirf wahi endpoints call karta hai jiski usko zaroorat hai.

### D - Dependency Inversion Principle (DIP)
**Concept:** High-level modules low-level modules par depend nahi hone chahiye. Dono abstractions par depend hone chahiye.
**How Sentinel Uses It:**
- `CleanPCPipeline` (Orchestrator): Jab ye initialize hota hai, toh isme `PlannerAgent`, `SafetyValidator`, aur `Executor` bahar se inject kiye jate hain (Dependency Injection). Isse testing (mocking) karna bahut aasan ho jata hai.

---

## 🎨 2. Design Patterns Used

Design patterns general, reusable solutions hote hain for commonly occurring problems in software design.

### 1. Chain of Responsibility (Pipeline Pattern)
**Kahan use hua:** `CleanPCPipeline` mein.
**Explanation:** Data ek pipeline me step-by-step pass hota hai:
1. `Scanner` data laata hai -> 
2. `FileClassifier` categorize karta hai -> 
3. `RulesEngine` rules lagata hai -> 
4. `AIPlanner` final plan banata hai -> 
5. `SafetyValidator` usko approve karta hai.
Agar koi bhi step fail hota hai, toh request wahi ruk jati hai ya fallback pe chali jati hai.

### 2. Command Pattern
**Kahan use hua:** `Executor` aur `PlanAction` mein.
**Explanation:** Command pattern request ko ek object me encapsulate kar deta hai. Sentinel me har ek move, rename, ya delete operation ek `PlanAction` object ban jata hai (e.g., `ActionType.MOVE`, source, destination). 
*Iska Fayda:* Iski wajah se hi Sentinel me **"Undo" (Reverse operation)** feature possible ho paya hai, kyunki har command ek object me logged rehti hai DB mein.

### 3. Strategy Pattern
**Kahan use hua:** AI Planning vs Fallback Planning mein.
**Explanation:** Strategy pattern runtime pe algorithm select karne ki azaadi deta hai.
`CleanPCPipeline` me agar `AIPlanner` kisi wajah se fail ho jaye, timeout ho jaye, ya Safety Validator plan reject kar de, toh system automatically apna algorithm change karke **Rule-based Fallback Strategy** use karta hai `_create_fallback_plan()` method ke through.

### 4. Facade Pattern
**Kahan use hua:** `SentinelAPI` (FastAPI layer) mein.
**Explanation:** Facade pattern ek complex system (Scanner, Planner, Executor, DB) ke aage ek simple interface de deta hai. 
Frontend (WebUI/Next.js) ko piche ke 10 classes aur unki complexity se koi matlab nahi. Frontend sirf `/api/scan-and-plan` hit karta hai. API ek facade ki tarah poore backend orchestration ko handle karti hai.

### 5. Observer (Pub/Sub) Pattern
**Kahan use hua:** WebSockets for Real-time Progress (`ws_manager` in Backend & `useWebSocket.ts` in Frontend).
**Explanation:** Ek object (Subject) state change hone par apne sabhi dependents (Observers) ko notify karta hai. 
Backend me jaise hi scanner ya executor ki progress badhti hai, WebSocket Manager us progress ko observe kar raha frontend (`useWebSocket` hook) par broadcast kar deta hai bina frontend ke baar-baar API poll kiye.

### 6. Singleton Pattern (Anti-Pattern managed well)
**Kahan use hua:** Database Connections aur WebSocket Manager mein.
**Explanation:** Poore application lifecycle mein `ws_manager` ka sirf ek hi global instance zinda rehta hai taaki saare active client connections ek hi jagah maintain rahein aur memory leak na ho.

---
**Conclusion:** 
Sentinel project ko ek standard "Script" ki tarah nahi, balki ek "Enterprise-grade System" ki tarah code kiya gaya hai. SOLID aur Design Patterns ka mixture is code ko easily readable, extendable, aur sabse important, **testable** banata hai.
