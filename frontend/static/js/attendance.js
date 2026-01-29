// frontend/static/js/attendance.js
// Handles webcam access, image capture, file upload, and API calls for attendance.

document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.getElementById('webcam-feed');
    const canvasElement = document.getElementById('snapshot-canvas');
    const context = canvasElement ? canvasElement.getContext('2d') : null;
    const recognizeButton = document.getElementById('recognize-button');
    const uploadInput = document.getElementById('upload-attendance-image');
    const uploadRecognizeButton = document.getElementById('upload-recognize-button');
    const resultsDiv = document.getElementById('recognition-results');
    const studentListDiv = document.getElementById('student-attendance-list');
    const webcamStatusElement = document.getElementById('webcam-status');
    
    // Get Session ID from the URL path (e.g., /take/1 -> 1)
    const currentSessionId = window.location.pathname.split('/').pop();
    let stream = null; 

    if (isNaN(parseInt(currentSessionId, 10))) {
        webcamStatusElement.textContent = 'ERROR: Invalid Session ID.';
        if(recognizeButton) recognizeButton.disabled = true;
        if(uploadRecognizeButton) uploadRecognizeButton.disabled = true;
        return;
    }

    // --- Webcam Setup ---
    async function setupWebcam() {
        if (!videoElement) return; 
        try {
            // Request video stream
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoElement.srcObject = stream;
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                recognizeButton.disabled = false;
                webcamStatusElement.textContent = 'Webcam ready.';
                webcamStatusElement.className = 'text-sm text-green-600 mt-2 text-center';
            };
        } catch (err) {
            console.error("Error accessing webcam:", err);
            recognizeButton.disabled = true;
            webcamStatusElement.textContent = `Error: ${err.name}. Check browser permissions.`;
            webcamStatusElement.className = 'text-sm text-red-600 mt-2 text-center';
        }
    }

    // --- Recognition API Call ---
    async function sendImageForRecognition(imageDataUrl, verificationMethod = 'FACE_LIVE') {
        resultsDiv.innerHTML = '<p class="text-blue-500 animate-pulse">Recognizing...</p>';
        
        try {
            const response = await fetch('/teacher/api/attendance/recognize', { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    image_data: imageDataUrl,
                    method: verificationMethod
                 }),
            });

            const result = await response.json();

            if (!response.ok) {
                // If Flask returned a non-200 status, the error is in result.error
                throw new Error(result.error || `Server error (Status: ${response.status})`);
            }

            displayRecognitionResults(result);
            updateStudentListStatus(result.recognized); 

        } catch (error) {
            console.error('API Call Error:', error);
            resultsDiv.innerHTML = `<p class="text-red-500 font-medium">Recognition Failed: ${error.message}</p>`;
        }
    }
    
    // --- Display Recognition Results ---
    function displayRecognitionResults(result) {
        let html = '<h3 class="text-lg font-semibold mb-2">Recognition Log:</h3><ul class="list-disc list-inside space-y-1">';
        let foundRecognized = false;

        if (result.recognized && result.recognized.length > 0) {
            result.recognized.forEach(student => {
                html += `<li class="text-green-600">Present: ${student.name} (${student.prn}) - Conf: ${student.confidence.toFixed(2)}</li>`;
                foundRecognized = true;
            });
        }
        if (result.unknown_count > 0) {
            html += `<li class="text-yellow-600">${result.unknown_count} Unknown face(s) detected/skipped.</li>`;
        }
        if (result.db_errors && result.db_errors.length > 0) {
            html += `<li class="text-red-600">DB Error(s): Failed to mark ${result.db_errors.length} record(s).</li>`;
        }

        if (!foundRecognized && result.unknown_count === 0 && result.marked_count === 0) {
             html += '<li class="text-gray-500">No known students recognized.</li>';
        }
        html += `<li class="mt-2 text-sm font-semibold">Marked Present: ${result.marked_count}</li></ul>`;
        resultsDiv.innerHTML = html;
    }

    // --- Update Student List UI ---
    function updateStudentListStatus(recognizedStudents) {
        if (!recognizedStudents || recognizedStudents.length === 0 || !studentListDiv) return;

        recognizedStudents.forEach(recStudent => {
            // Target the row using the data attribute
            const studentRow = studentListDiv.querySelector(`[data-student-id="${recStudent.student_id}"]`); 
            if (studentRow) {
                const statusSpan = studentRow.querySelector('.attendance-status');
                const presentBtn = studentRow.querySelector('[data-status="Present"]');
                const absentBtn = studentRow.querySelector('[data-status="Absent"]');
                
                if (statusSpan) {
                    statusSpan.textContent = 'Present';
                    statusSpan.className = 'attendance-status text-sm font-semibold text-green-600';
                }
                
                // Update manual button appearance to show 'Present' is active
                if (presentBtn) {
                    presentBtn.classList.add('bg-green-200', 'text-green-800');
                    presentBtn.classList.remove('bg-green-500', 'text-white', 'hover:bg-green-600');
                }
                 if (absentBtn) {
                    absentBtn.classList.remove('bg-red-200', 'text-red-800');
                    absentBtn.classList.add('bg-red-500', 'text-white', 'hover:bg-red-600');
                }
            }
        });
    }

    // --- Event Handlers ---

    // 1. Capture from Webcam
    if (recognizeButton) {
        recognizeButton.addEventListener('click', () => {
            if (!videoElement || !context || videoElement.readyState < 3) {
                webcamStatusElement.textContent = `Webcam not ready. Status: ${videoElement.readyState}`;
                return;
            }
            try {
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
                const imageDataUrl = canvasElement.toDataURL('image/jpeg', 0.9);
                
                recognizeButton.disabled = true;
                recognizeButton.textContent = 'Processing...';

                sendImageForRecognition(imageDataUrl, 'FACE_LIVE').finally(() => {
                    recognizeButton.disabled = false;
                    recognizeButton.textContent = 'Capture & Recognize Frame';
                });

            } catch (e) {
                 console.error("Error capturing frame:", e);
                 resultsDiv.innerHTML = `<p class="text-red-500">Error capturing frame from webcam.</p>`;
            }
        });
    }

    // 2. Recognize from Uploaded Image(s)
     if (uploadRecognizeButton && uploadInput) {
        uploadRecognizeButton.addEventListener('click', () => {
            const files = uploadInput.files;
            if (files.length === 0) {
                resultsDiv.innerHTML = '<p class="text-yellow-600">Please select one or more image files first.</p>';
                return;
            }
            
            uploadRecognizeButton.disabled = true;
            uploadRecognizeButton.textContent = 'Processing...';
            resultsDiv.innerHTML = '<p class="text-blue-500 animate-pulse">Processing uploaded image(s)...</p>';

            // We only send the first file for simplicity in this synchronous API, 
            // but a real solution would iterate through all files (like in student_logic)
            const file = files[0]; 
            const reader = new FileReader();
            
            reader.onload = function(event) {
                const imageDataUrl = event.target.result;
                sendImageForRecognition(imageDataUrl, 'FACE_UPLOAD')
                    .finally(() => {
                        uploadRecognizeButton.disabled = false;
                        uploadRecognizeButton.textContent = 'Upload & Recognize Image(s)';
                        uploadInput.value = ''; // Clear input
                    });
            };
            reader.readAsDataURL(file);
        });
     }

    // 3. Manual Marking (using event delegation on the list)
     if (studentListDiv) {
        studentListDiv.addEventListener('click', (event) => {
            if (event.target.classList.contains('manual-mark-btn')) {
                const button = event.target;
                const studentId = button.dataset.studentId;
                const newStatus = button.dataset.status; 
                
                if (!studentId || !newStatus || !currentSessionId) return;

                // Disable all buttons in this row temporarily
                studentListDiv.querySelector(`[data-student-id="${studentId}"]`).querySelectorAll('button').forEach(btn => btn.disabled = true);
                
                fetch('/teacher/api/attendance/mark_manual', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: currentSessionId,
                        student_id: parseInt(studentId, 10),
                        status: newStatus
                    }),
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const row = studentListDiv.querySelector(`[data-student-id="${studentId}"]`);
                        if (row) {
                            const statusSpan = row.querySelector('.attendance-status');
                            statusSpan.textContent = newStatus;
                            statusSpan.className = `attendance-status text-sm font-semibold ${
                                newStatus === 'Present' ? 'text-green-600' : (newStatus === 'Late' ? 'text-yellow-600' : 'text-red-600')
                            }`;
                            
                            // Highlight the active button
                            row.querySelectorAll('button').forEach(btn => {
                                btn.disabled = false; // Re-enable
                                btn.classList.remove('bg-green-200', 'bg-red-200', 'text-green-800', 'text-red-800');
                                if (btn.dataset.status === newStatus) {
                                    btn.classList.add(`bg-${newStatus.toLowerCase()}-200`, `text-${newStatus.toLowerCase()}-800`);
                                    btn.classList.remove(`bg-${newStatus.toLowerCase()}-500`, 'text-white', `hover:bg-${newStatus.toLowerCase()}-600`);
                                } else {
                                    // Reset inactive buttons to default color
                                    const statusColor = btn.dataset.status.toLowerCase();
                                    btn.classList.add(`bg-${statusColor}-500`, 'text-white', `hover:bg-${statusColor}-600`);
                                }
                            });
                        }
                    } else {
                         throw new Error(data.error || 'Manual marking failed.');
                    }
                })
                .catch(error => {
                    console.error('Error manual marking:', error);
                    alert(`Error manual marking: ${error.message}`); 
                    studentListDiv.querySelector(`[data-student-id="${studentId}"]`).querySelectorAll('button').forEach(btn => btn.disabled = false); // Re-enable on error
                });
            }
        });
     }


    // Initialize webcam when the page loads
    setupWebcam();

    // Cleanup: Stop webcam stream when navigating away
    window.addEventListener('beforeunload', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });
});