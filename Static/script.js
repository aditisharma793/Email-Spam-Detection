const messageInput = document.getElementById("message");

const checkButton = document.getElementById("checkButton");

const result = document.getElementById("result");

checkButton.addEventListener("click", async function() {

    const message = messageInput.value.trim();

    // Check empty input
    if (message === "") {

        result.textContent = "Please enter a message.";

        result.style.color = "red";

        return;
    }

    // Show loading message
    result.textContent = "Checking your message...";

    result.style.color = "#2563eb";

    try {

        // Send message to Flask backend
        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: message
            })

        });

        const data = await response.json();

        // Handle errors
        if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
        }

        // Display actual ML prediction
        if (data.prediction === "SPAM") {

            result.textContent = "⚠ SPAM MESSAGE DETECTED!";

            result.style.color = "red";

        } else {

            result.textContent = "✓ THIS IS NOT A SPAM MESSAGE.";

            result.style.color = "green";

        }

    } catch (error) {

        result.textContent = error.message;

        result.style.color = "red";

    }

});