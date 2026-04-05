document.addEventListener("DOMContentLoaded", () => {
    const flashMessages = document.querySelectorAll("[data-flash]");
    const footerStatus = document.getElementById("footer-status");

    flashMessages.forEach((message) => {
        const closeButton = message.querySelector("[data-flash-close]");

        if (closeButton) {
            closeButton.addEventListener("click", () => {
                message.remove();
            });
        }
    });

    if (footerStatus) {
        footerStatus.textContent = "Aplicacao carregada e pronta para receber as proximas evolucoes.";
    }
});
