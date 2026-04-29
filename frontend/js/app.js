// ================= TEXT EMOTION =================
document.getElementById("textBtn").addEventListener("click", async () => {
    const text = document.getElementById("textInput").value;

    if (!text.trim()) {
        alert("Please enter text");
        return;
    }

    try {
        const response = await fetch("http://127.0.0.1:5000/predict-text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        document.getElementById("textEmotion").innerText =
            "Emotion: " + data.emotion;

        document.getElementById("textConfidence").innerText =
            "Confidence: " + data.confidence + "%";

    } catch (err) {
        alert("Text Emotion: Server error");
        console.error(err);
    }
});

// ================= FACE EMOTION =================
document.getElementById("faceBtn").addEventListener("click", async () => {
    const fileInput = document.getElementById("imageInput");
    if (!fileInput.files.length) {
        alert("Please select an image");
        return;
    }

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    try {
        const response = await fetch("http://127.0.0.1:5000/predict-face", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        document.getElementById("faceEmotion").innerText =
            "Emotion: " + data.emotion;

        document.getElementById("faceConfidence").innerText =
            "Confidence: " + data.confidence + "%";

    } catch (err) {
        alert("Face Emotion: Server error");
        console.error(err);
    }
});
