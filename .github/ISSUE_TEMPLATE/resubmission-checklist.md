# 🚨 Resubmission Checklist

---

## 🔴 Critical Issues (Must Fix)

### #1 Model loaded on every request
- [ ] Move model loading outside the API endpoint. Load model once at startup or globally

**Why this matters:**
Loading the model inside the endpoint causes heavy latency on every request.  
This leads to extremely slow responses (seconds to minutes).  
It also increases CPU usage unnecessarily.  
The model should be loaded once and reused for performance.

---

### #2 API does not return JSON response
- [ ] Update endpoint to return JSON format. Replace raw string output with structured response

**Why this matters:**
APIs must return consistent structured data for clients.  
Raw strings break frontend integration and API usability.  
JSON ensures compatibility with web apps and tools.  
Standard format: `{"impression": result}`.

---

### #3 Missing error handling in inference
- [ ] Add try/except around model inference. Return safe error messages on failure

**Why this matters:**
Without error handling, any invalid input will crash the API.  
This makes the system unreliable in production.  
Proper handling ensures graceful failure instead of downtime.  
It improves stability and user experience.

---

### #4 Unsafe model path (deployment fragile)
- [ ] Replace relative path (`../models/...`). Use OS-safe path with `os.path` and `__file__`

**Why this matters:**
Relative paths depend on where the server is started from.  
On Render, working directories are not guaranteed.  
This can cause FileNotFoundError in production.  
Using `__file__` ensures stable path resolution everywhere.

---
### #5 Blocking async endpoint (performance bottleneck)
- [ ] Change endpoint to `def` to prevent the app from 'freezing'.

**Why this matters:**
PyTorch inference is CPU blocking and should not run directly inside async endpoints.
It blocks the event loop and reduces concurrent request handling.

---

## 🟠 Deployment Issues (Render Free Tier)

### #6 Incorrect dependency setup for production
- [ ] Separate training and deployment requirements files. Create minimal production requirements file

**Why this matters:**
Training libraries are heavy and unnecessary in production.  
They slow down deployment and increase build failure risk.  
Render should only install runtime dependencies.  
This keeps the app lightweight and stable.


---


## 🟡 Performance Issues

### #7 No inference optimization
- [ ] Add `model.eval()`
- [ ] Wrap inference in `torch.no_grad()`

**Why this matters:**
Prevents unnecessary gradient tracking during inference.  
Improves speed and reduces memory usage.  
Ensures model runs in evaluation mode.  
Standard practice for ML deployment.

---

### #8 No input validation
- [ ] Add max input length restriction.Prevent oversized requests

**Why this matters:**
Large inputs can crash or slow down inference.  
They may cause memory spikes or timeouts.  
Validation ensures stable performance.  
It protects the API from misuse.

---


## 🟢 Optional 


### #9 Add simple HTML test interface
- [ ] Create input box UI. Display model response and api usage info.

**Why this matters:**
Makes testing easier without Postman.  
Useful for demos and portfolios.  
Improves user experience.
