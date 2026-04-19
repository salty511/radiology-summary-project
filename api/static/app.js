const form = document.getElementById("impression-form");
const findingsInput = document.getElementById("findings");
const submitBtn = document.getElementById("submit-btn");
const resultBox = document.getElementById("result");
const statusText = document.getElementById("status");
const charCount = document.getElementById("char-count");

console.log("JS LOADED")

findingsInput.addEventListener("input", () => {
  charCount.textContent = findingsInput.value.length;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const findingsText = findingsInput.value.trim();

  submitBtn.disabled = true;
  statusText.textContent = "Generating...";
  resultBox.hidden = true;
  resultBox.classList.remove("error");

  try {
    const impressionURL = "/impression/"
    console.log(impressionURL)
    const response = await fetch(impressionURL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ findingsText })
    });
    console.log(response.status)
    const data = await response.json();
  
    if (!response.ok || !data.impression) {
      throw new Error(data.detail || data.exception || "Request failed");
    }

    resultBox.textContent = data.impression;
    resultBox.hidden = false;
    statusText.textContent = "Done";
  } catch (error) {
    resultBox.textContent = "Error: " + error.message;
    resultBox.classList.add("error");
    resultBox.hidden = false;
    statusText.textContent = "Request failed";
  } finally {
    submitBtn.disabled = false;
  }
});
