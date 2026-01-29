document.addEventListener('DOMContentLoaded', function () {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    })

    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const spinner = submitBtn.querySelector('.spinner-border');
    const resetBtn = document.getElementById('resetBtn');

    const resultSection = document.getElementById('resultSection');
    const resultCard = document.getElementById('resultCard');
    const resultIcon = document.getElementById('resultIcon');
    const predictionText = document.getElementById('predictionText');
    const probabilityText = document.getElementById('probabilityText');
    const probabilityBar = document.getElementById('probabilityBar');

    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');

    // Handle Form Submission
    form.addEventListener('submit', function (e) {
        e.preventDefault();

        // Hide previous results/errors
        resultSection.classList.add('d-none');
        errorAlert.classList.add('d-none');

        // Show loading state
        submitBtn.disabled = true;
        spinner.classList.remove('d-none');

        // Collect data
        // Collect data
        const formData = {
            FastingGlucose: document.getElementById('FastingGlucose').value,
            HbA1c: document.getElementById('HbA1c').value,
            OGTT_2hr: document.getElementById('OGTT_2hr').value,
            FastingInsulin: document.getElementById('FastingInsulin').value,
            HOMA_IR: document.getElementById('HOMA_IR').value,
            BMI: document.getElementById('BMI').value,
            WaistCircumference: document.getElementById('WaistCircumference').value,
            WaistHipRatio: document.getElementById('WaistHipRatio').value,
            SystolicBP: document.getElementById('SystolicBP').value,
            Triglycerides: document.getElementById('Triglycerides').value,
            HDL: document.getElementById('HDL').value,
            Age: document.getElementById('Age').value,
            FamilyHistory: document.getElementById('FamilyHistory').value,
            PhysicalActivity: document.getElementById('PhysicalActivity').value,
            Sex: document.getElementById('Sex').value
        };

        // Send API Request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.error || 'Server Error'); });
                }
                return response.json();
            })
            .then(data => {
                displayResult(data);
            })
            .catch(error => {
                showError(error.message);
            })
            .finally(() => {
                // Reset loading state
                submitBtn.disabled = false;
                spinner.classList.add('d-none');
            });
    });

    // Handle Reset
    resetBtn.addEventListener('click', function () {
        resultSection.classList.add('d-none');
        errorAlert.classList.add('d-none');
        form.reset();
    });

    function displayResult(data) {
        resultSection.classList.remove('d-none');

        const isDiabetic = data.prediction === 'Diabetic';
        const prob = data.probability;

        // Update Text
        predictionText.textContent = data.prediction.toUpperCase();
        probabilityText.textContent = prob + '%';

        // Update Progress Bar
        probabilityBar.style.width = prob + '%';
        probabilityBar.textContent = prob + '%';

        // Style based on result
        if (isDiabetic) {
            // Diabetic Styling (Red)
            resultCard.classList.remove('card-success');
            resultCard.classList.add('card-danger');

            resultIcon.className = 'fas fa-exclamation-circle text-danger';
            predictionText.className = 'fw-bold mb-3 display-5 text-danger';

            probabilityBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-danger fw-bold fs-6';
        } else {
            // Non-Diabetic Styling (Green)
            resultCard.classList.remove('card-danger');
            resultCard.classList.add('card-success');

            resultIcon.className = 'fas fa-check-circle text-success';
            predictionText.className = 'fw-bold mb-3 display-5 text-success';

            probabilityBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-success fw-bold fs-6';
        }

        // Scroll to result
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorAlert.classList.remove('d-none');
    }
});
