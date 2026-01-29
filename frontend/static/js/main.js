console.log("Attendify Main JS Loaded");

// Example: Auto-hide flash messages after a few seconds
document.addEventListener('DOMContentLoaded', (event) => {
    const flashMessages = document.querySelectorAll('.flash-message'); // Add a common class to your flash message divs
    flashMessages.forEach(flashMessage => {
        setTimeout(() => {
            flashMessage.style.transition = 'opacity 0.5s ease';
            flashMessage.style.opacity = '0';
            setTimeout(() => {
                 // flashMessage.remove(); // Remove from DOM after fading
                 flashMessage.style.display = 'none'; // Or just hide
            }, 500); // Wait for fade out
        }, 5000); // Hide after 5 seconds
    });

     // Add confirmation for delete actions (example)
    const deleteButtons = document.querySelectorAll('.delete-button'); // Add this class to delete links/buttons
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            if (!confirm('Are you sure you want to delete this item? This action cannot be undone.')) {
                e.preventDefault(); // Prevent the default action (e.g., following a link or submitting a form)
            }
        });
    });
});

// Add other global JS functions here if needed