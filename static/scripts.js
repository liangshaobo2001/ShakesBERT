document.getElementById("submit-button").addEventListener("click", function () {
    this.textContent = "Submitting...";
    setTimeout(() => {
        this.textContent = "Submit";
    }, 2000);
});
