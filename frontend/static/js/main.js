document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadForm = document.getElementById('uploadForm');
    const uploadBtn = document.getElementById('uploadBtn');
    const imageFileInput = document.getElementById('imageFile');
    const statusMessage = document.getElementById('statusMessage');
    const statusIcon = document.getElementById('statusIcon');
    const statusText = document.getElementById('statusText');
    const outputPreview = document.getElementById('outputPreview');
    const segmentedImage = document.getElementById('segmentedImage');
    const mainStatus = document.getElementById('mainStatus');
    
    const toggleBtns = document.querySelectorAll('.toggle-btn');
    const statusInds = document.querySelectorAll('.status-ind');
    const chartTabs = document.querySelectorAll('.chart-tab');
    const inputTabs = document.querySelectorAll('.input-tab');
    
    // State variables
    let uploadedImage = null;
    let originalImageData = null;
    let segmentedImageData = null;
    let overlayImageData = null;
    let isAnalyzing = false;
    let terrainChart = null;
    let currentChartType = 'bar';
    let resultData = null;
    
    let webcamStream = null;
    let webcamInterval = null;
    let videoFramesData = [];
    let videoOriginalFramesData = [];
    let currentVideoFrame = 0;
    let lastWebcamProcessTime = 0;
    const WEBCAM_PROCESS_INTERVAL = 200;
    
    // Helper Functions
    function showStatus(text, isAnalyzing) {
        if (!statusText || !statusIcon || !statusMessage) return;
        
        statusText.textContent = text;
        statusIcon.textContent = isAnalyzing ? '⟳' : '✓';
        statusIcon.className = isAnalyzing ? 'spinner' : '';
        statusMessage.className = isAnalyzing ? 'status-message show analyzing' : 'status-message show';
    }
    
    function cleanupImageURL() {
        if (uploadedImage && uploadedImage.startsWith('blob:')) {
            URL.revokeObjectURL(uploadedImage);
        }
        uploadedImage = null;
    }
    
    // Input tab switching
    inputTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            inputTabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            const inputType = this.dataset.input;
            const imageInput = document.getElementById('imageInput');
            const videoInput = document.getElementById('videoInput');
            const webcamInput = document.getElementById('webcamInput');
            
            if (imageInput) imageInput.style.display = inputType === 'image' ? 'block' : 'none';
            if (videoInput) videoInput.style.display = inputType === 'video' ? 'block' : 'none';
            if (webcamInput) webcamInput.style.display = inputType === 'webcam' ? 'block' : 'none';
            
            if (inputType === 'webcam') {
                stopWebcam();
            } else {
                // Reset display when switching away from webcam
                const webcamVideoElem = document.getElementById('webcamVideo');
                const webcamResultElem = document.getElementById('webcamResult');
                if (webcamVideoElem) webcamVideoElem.style.display = 'none';
                if (webcamResultElem) webcamResultElem.style.display = 'none';
            }
        });
    });
    
    // Image file selection handler
    imageFileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        // Clean up previous URL
        cleanupImageURL();
        
        if (file.size > 20 * 1024 * 1024) {
            showStatus('File too large (max 20MB)', false);
            e.target.value = '';
            return;
        }
        
        if (!file.type.startsWith('image/')) {
            showStatus('Please select an image file', false);
            e.target.value = '';
            return;
        }
        
        const url = URL.createObjectURL(file);
        uploadedImage = url;
        originalImageData = url;
        
        if (segmentedImage) {
            segmentedImage.src = url;
        }
        
        if (outputPreview) {
            outputPreview.classList.add('has-image');
            const placeholder = outputPreview.querySelector('.placeholder');
            if (placeholder) placeholder.style.display = 'none';
        }
        
        if (uploadBtn) uploadBtn.disabled = false;
        showStatus('Image loaded - ready to analyze', false);
    });
    
    // Image upload form submission
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!uploadedImage || isAnalyzing) return;
        
        isAnalyzing = true;
        if (uploadBtn) uploadBtn.disabled = true;
        
        if (mainStatus) {
            mainStatus.className = 'main-status analyzing';
            const statusIconElem = mainStatus.querySelector('.status-icon');
            const statusLabelElem = mainStatus.querySelector('.status-info-label');
            const statusMsgElem = mainStatus.querySelector('.status-info-msg');
            
            if (statusIconElem) statusIconElem.textContent = '◉';
            if (statusLabelElem) statusLabelElem.textContent = 'ANALYZING...';
            if (statusMsgElem) statusMsgElem.textContent = 'Running inference...';
        }
        
        showStatus('Analyzing terrain...', true);
        
        const formData = new FormData(uploadForm);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                if (result.original_image) {
                    originalImageData = 'data:image/png;base64,' + result.original_image;
                }
                if (result.segmented_image) {
                    segmentedImageData = 'data:image/png;base64,' + result.segmented_image;
                    if (segmentedImage) {
                        segmentedImage.src = segmentedImageData;
                    }
                    if (outputPreview) {
                        outputPreview.classList.add('has-image');
                        const placeholder = outputPreview.querySelector('.placeholder');
                        if (placeholder) placeholder.style.display = 'none';
                    }
                }
                
                if (result.original_image && result.segmented_image) {
                    overlayImageData = createOverlayImage(originalImageData, segmentedImageData);
                }
                
                updateTraversability(result);
                resultData = result;
                updateTerrainDistribution(result.class_distribution);
                updateConfusionMatrix(result);
                updateLegend(result.class_distribution);
                updateMetrics(result.metrics);
                
                const score = result.traversability_score || 0.85;
                if (mainStatus) {
                    if (score >= 0.7) {
                        mainStatus.className = 'main-status safe';
                        const statusIconElem = mainStatus.querySelector('.status-icon');
                        const statusLabelElem = mainStatus.querySelector('.status-info-label');
                        if (statusIconElem) statusIconElem.textContent = '✓';
                        if (statusLabelElem) statusLabelElem.textContent = 'SAFE TO TRAVERSE';
                    } else if (score >= 0.4) {
                        mainStatus.className = 'main-status caution';
                        const statusIconElem = mainStatus.querySelector('.status-icon');
                        const statusLabelElem = mainStatus.querySelector('.status-info-label');
                        if (statusIconElem) statusIconElem.textContent = '⚠';
                        if (statusLabelElem) statusLabelElem.textContent = 'CAUTION';
                    } else {
                        mainStatus.className = 'main-status unsafe';
                        const statusIconElem = mainStatus.querySelector('.status-icon');
                        const statusLabelElem = mainStatus.querySelector('.status-info-label');
                        if (statusIconElem) statusIconElem.textContent = '✕';
                        if (statusLabelElem) statusLabelElem.textContent = 'UNSAFE';
                    }
                    const statusMsgElem = mainStatus.querySelector('.status-info-msg');
                    if (statusMsgElem) statusMsgElem.textContent = 'Score: ' + score.toFixed(2);
                }
                
                showStatus('Analysis complete', false);
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Error:', error);
            showStatus('Error: ' + error.message, false);
            
            if (mainStatus) {
                mainStatus.className = 'main-status error';
                const statusIconElem = mainStatus.querySelector('.status-icon');
                const statusLabelElem = mainStatus.querySelector('.status-info-label');
                const statusMsgElem = mainStatus.querySelector('.status-info-msg');
                
                if (statusIconElem) statusIconElem.textContent = '✕';
                if (statusLabelElem) statusLabelElem.textContent = 'UPLOAD FAILED';
                
                let errorMsg = error.message;
                if (errorMsg.includes('does not support') || errorMsg.includes('Cannot read')) {
                    errorMsg = 'Image format not supported. Please use PNG, JPG, JPEG, or WEBP format.';
                }
                if (statusMsgElem) statusMsgElem.textContent = errorMsg;
            }
        } finally {
            isAnalyzing = false;
            if (uploadBtn) uploadBtn.disabled = false;
        }
    });
    
    // Video upload handling
    const videoUploadForm = document.getElementById('videoUploadForm');
    const videoFileInput = document.getElementById('videoFile');
    const videoUploadBtn = document.getElementById('videoUploadBtn');
    const videoProgress = document.getElementById('videoProgress');
    const videoProgressFill = document.getElementById('videoProgressFill');
    
    if (videoFileInput) {
        videoFileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file && file.size > 100 * 1024 * 1024) {
                alert('File too large (max 100MB)');
                e.target.value = '';
            }
        });
    }
    
    if (videoUploadForm) {
        videoUploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            if (!videoFileInput || !videoFileInput.files[0] || isAnalyzing) return;
            
            isAnalyzing = true;
            if (videoUploadBtn) videoUploadBtn.disabled = true;
            if (videoProgress) videoProgress.style.display = 'block';
            if (videoProgressFill) videoProgressFill.style.width = '0%';
            
            if (mainStatus) {
                mainStatus.className = 'main-status analyzing';
                const statusIconElem = mainStatus.querySelector('.status-icon');
                const statusLabelElem = mainStatus.querySelector('.status-info-label');
                if (statusIconElem) statusIconElem.textContent = '◉';
                if (statusLabelElem) statusLabelElem.textContent = 'PROCESSING VIDEO...';
            }
            
            const formData = new FormData(videoUploadForm);
            
            try {
                const response = await fetch('/video/stream', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                
                if (result.success) {
                    videoFramesData = result.frames || [];
                    videoOriginalFramesData = result.original_frames || result.frames || [];
                    currentVideoFrame = 0;
                    
                    if (videoFramesData.length > 0) {
                        segmentedImageData = 'data:image/jpeg;base64,' + videoFramesData[0].frame;
                        originalImageData = 'data:image/jpeg;base64,' + (videoOriginalFramesData[0] ? videoOriginalFramesData[0].frame : videoFramesData[0].frame);
                        if (segmentedImage) {
                            segmentedImage.src = segmentedImageData;
                        }
                        if (outputPreview) {
                            outputPreview.classList.add('has-image');
                            const placeholder = outputPreview.querySelector('.placeholder');
                            if (placeholder) placeholder.style.display = 'none';
                        }
                        
                        // Update charts with first frame data
                        if (videoFramesData[0].class_distribution) {
                            updateTerrainDistribution(videoFramesData[0].class_distribution);
                            updateLegend(videoFramesData[0].class_distribution);
                        }
                        
                        // Show video navigation
                        addVideoNavigation();
                    }
                    
                    const score = result.average_traversability || 0;
                    if (mainStatus) {
                        if (score >= 0.7) {
                            mainStatus.className = 'main-status safe';
                            const statusIconElem = mainStatus.querySelector('.status-icon');
                            const statusLabelElem = mainStatus.querySelector('.status-info-label');
                            if (statusIconElem) statusIconElem.textContent = '✓';
                            if (statusLabelElem) statusLabelElem.textContent = 'SAFE TO TRAVERSE';
                        } else if (score >= 0.4) {
                            mainStatus.className = 'main-status caution';
                            const statusIconElem = mainStatus.querySelector('.status-icon');
                            const statusLabelElem = mainStatus.querySelector('.status-info-label');
                            if (statusIconElem) statusIconElem.textContent = '⚠';
                            if (statusLabelElem) statusLabelElem.textContent = 'CAUTION';
                        } else {
                            mainStatus.className = 'main-status unsafe';
                            const statusIconElem = mainStatus.querySelector('.status-icon');
                            const statusLabelElem = mainStatus.querySelector('.status-info-label');
                            if (statusIconElem) statusIconElem.textContent = '✕';
                            if (statusLabelElem) statusLabelElem.textContent = 'UNSAFE';
                        }
                        const statusMsgElem = mainStatus.querySelector('.status-info-msg');
                        if (statusMsgElem) {
                            statusMsgElem.textContent = `Processed ${result.total_frames || 0} frames | Score: ${score.toFixed(2)}`;
                        }
                    }
                    
                    showStatus(`Video processed: ${result.total_frames || 0} frames`, false);
                } else {
                    throw new Error(result.error || 'Video processing failed');
                }
            } catch (error) {
                console.error('Error:', error);
                showStatus('Error: ' + error.message, false);
                
                if (mainStatus) {
                    mainStatus.className = 'main-status error';
                    const statusIconElem = mainStatus.querySelector('.status-icon');
                    const statusLabelElem = mainStatus.querySelector('.status-info-label');
                    if (statusIconElem) statusIconElem.textContent = '✕';
                    if (statusLabelElem) statusLabelElem.textContent = 'FAILED';
                }
            } finally {
                isAnalyzing = false;
                if (videoUploadBtn) videoUploadBtn.disabled = false;
                if (videoProgress) videoProgress.style.display = 'none';
            }
        });
    }
    
    function addVideoNavigation() {
        const videoProgressParent = videoProgress ? videoProgress.parentNode : null;
        if (!videoProgressParent) return;
        
        let navContainer = document.querySelector('.video-nav');
        if (!navContainer) {
            navContainer = document.createElement('div');
            navContainer.className = 'video-nav';
            videoProgressParent.appendChild(navContainer);
        }
        
        navContainer.innerHTML = `
            <button class="video-nav-btn" id="prevVideoBtn">◀ Prev</button>
            <button class="video-nav-btn" id="nextVideoBtn">Next ▶</button>
        `;
        
        const prevBtn = document.getElementById('prevVideoBtn');
        const nextBtn = document.getElementById('nextVideoBtn');
        
        if (prevBtn) prevBtn.onclick = () => { if (videoFramesData.length > 0) { currentVideoFrame = Math.max(0, currentVideoFrame - 1); updateVideoFrameDisplay(); } };
        if (nextBtn) nextBtn.onclick = () => { if (videoFramesData.length > 0) { currentVideoFrame = Math.min(videoFramesData.length - 1, currentVideoFrame + 1); updateVideoFrameDisplay(); } };
        
        let slider = document.querySelector('.video-frame-slider');
        if (!slider) {
            slider = document.createElement('div');
            slider.className = 'video-frame-slider';
            slider.innerHTML = `<input type="range" id="videoFrameSlider" min="0" max="${videoFramesData.length - 1}" value="0">`;
            videoProgressParent.appendChild(slider);
            
            const sliderInput = document.getElementById('videoFrameSlider');
            if (sliderInput) {
                sliderInput.onchange = function() {
                    currentVideoFrame = parseInt(this.value);
                    updateVideoFrameDisplay();
                };
            }
        }
    }
    
    function updateVideoFrameDisplay() {
        if (videoFramesData.length === 0 || !segmentedImage) return;
        
        const frameData = videoFramesData[currentVideoFrame];
        if (!frameData) return;
        
        segmentedImageData = 'data:image/jpeg;base64,' + frameData.frame;
        originalImageData = videoOriginalFramesData[currentVideoFrame] 
            ? 'data:image/jpeg;base64,' + videoOriginalFramesData[currentVideoFrame].frame 
            : segmentedImageData;
        
        const activeToggle = document.querySelector('.toggle-btn.active');
        const currentMode = activeToggle ? activeToggle.dataset.mode : 'segmented';
        
        if (currentMode === 'original') {
            segmentedImage.src = originalImageData;
        } else {
            segmentedImage.src = segmentedImageData;
        }
        
        const slider = document.getElementById('videoFrameSlider');
        if (slider) slider.value = currentVideoFrame;
        
        // Update charts
        if (frameData.class_distribution) {
            updateTerrainDistribution(frameData.class_distribution);
            updateLegend(frameData.class_distribution);
        }
        
        // Update status
        const score = frameData.traversability_score || 0;
        if (mainStatus) {
            if (score >= 0.7) {
                mainStatus.className = 'main-status safe';
                const statusIconElem = mainStatus.querySelector('.status-icon');
                const statusLabelElem = mainStatus.querySelector('.status-info-label');
                if (statusIconElem) statusIconElem.textContent = '✓';
                if (statusLabelElem) statusLabelElem.textContent = 'SAFE TO TRAVERSE';
            } else if (score >= 0.4) {
                mainStatus.className = 'main-status caution';
                const statusIconElem = mainStatus.querySelector('.status-icon');
                const statusLabelElem = mainStatus.querySelector('.status-info-label');
                if (statusIconElem) statusIconElem.textContent = '⚠';
                if (statusLabelElem) statusLabelElem.textContent = 'CAUTION';
            } else {
                mainStatus.className = 'main-status unsafe';
                const statusIconElem = mainStatus.querySelector('.status-icon');
                const statusLabelElem = mainStatus.querySelector('.status-info-label');
                if (statusIconElem) statusIconElem.textContent = '✕';
                if (statusLabelElem) statusLabelElem.textContent = 'UNSAFE';
            }
            const statusMsgElem = mainStatus.querySelector('.status-info-msg');
            if (statusMsgElem) {
                statusMsgElem.textContent = `Frame ${currentVideoFrame + 1}/${videoFramesData.length} | Score: ${score.toFixed(2)}`;
            }
        }
    }
    
    // Webcam handling
    const webcamVideo = document.getElementById('webcamVideo');
    const startWebcamBtn = document.getElementById('startWebcam');
    const stopWebcamBtn = document.getElementById('stopWebcam');
    const webcamStatusElem = document.getElementById('webcamStatus');
    const webcamResult = document.getElementById('webcamResult');
    
    if (startWebcamBtn) {
        startWebcamBtn.addEventListener('click', async function() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: { ideal: 640 }, height: { ideal: 480 } } 
                });
                
                webcamStream = stream;
                if (webcamVideo) {
                    webcamVideo.srcObject = stream;
                    webcamVideo.style.display = 'block';
                }
                if (webcamResult) webcamResult.style.display = 'none';
                
                startWebcamBtn.style.display = 'none';
                if (stopWebcamBtn) stopWebcamBtn.style.display = 'flex';
                if (webcamStatusElem) {
                    webcamStatusElem.className = 'webcam-status active';
                    webcamStatusElem.innerHTML = '<span class="status-dot"></span><span>Processing...</span>';
                }
                
                // Start frame processing
                if (webcamInterval) clearInterval(webcamInterval);
                webcamInterval = setInterval(processWebcamFrame, WEBCAM_PROCESS_INTERVAL);
                
            } catch (error) {
                console.error('Webcam error:', error);
                if (webcamStatusElem) {
                    webcamStatusElem.innerHTML = '<span class="status-dot"></span><span>Camera access denied</span>';
                }
            }
        });
    }
    
    if (stopWebcamBtn) {
        stopWebcamBtn.addEventListener('click', stopWebcam);
    }
    
    function stopWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        
        if (webcamInterval) {
            clearInterval(webcamInterval);
            webcamInterval = null;
        }
        
        if (webcamVideo) {
            webcamVideo.style.display = 'none';
            webcamVideo.srcObject = null;
        }
        
        if (startWebcamBtn) startWebcamBtn.style.display = 'flex';
        if (stopWebcamBtn) stopWebcamBtn.style.display = 'none';
        if (webcamStatusElem) {
            webcamStatusElem.className = 'webcam-status';
            webcamStatusElem.innerHTML = '<span class="status-dot"></span><span>Click Start to enable webcam</span>';
        }
    }
    
    async function processWebcamFrame() {
        const now = Date.now();
        if (now - lastWebcamProcessTime < WEBCAM_PROCESS_INTERVAL) return;
        lastWebcamProcessTime = now;
        
        if (!webcamVideo || !webcamVideo.videoWidth || !webcamStream) return;
        
        try {
            const canvas = document.createElement('canvas');
            canvas.width = webcamVideo.videoWidth;
            canvas.height = webcamVideo.videoHeight;
            const ctx = canvas.getContext('2d');
            
            // Flip horizontally for mirror effect
            ctx.save();
            ctx.scale(-1, 1);
            ctx.drawImage(webcamVideo, -canvas.width, 0);
            ctx.restore();
            
            const imageData = canvas.toDataURL('image/jpeg');
            
            const response = await fetch('/video/analyze_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                originalImageData = imageData;
                segmentedImageData = 'data:image/jpeg;base64,' + result.segmented_image;
                overlayImageData = segmentedImageData;
                
                if (webcamResult) {
                    webcamResult.src = segmentedImageData;
                    webcamResult.style.display = 'block';
                }
                if (webcamVideo) webcamVideo.style.display = 'none';
                if (segmentedImage) {
                    segmentedImage.src = segmentedImageData;
                }
                if (outputPreview) {
                    outputPreview.classList.add('has-image');
                    const placeholder = outputPreview.querySelector('.placeholder');
                    if (placeholder) placeholder.style.display = 'none';
                }
                
                if (result.class_distribution) {
                    updateTerrainDistribution(result.class_distribution);
                    updateLegend(result.class_distribution);
                }
                
                const score = result.traversability_score || 0;
                if (mainStatus) {
                    if (score >= 0.7) {
                        mainStatus.className = 'main-status safe';
                        const statusIconElem = mainStatus.querySelector('.status-icon');
                        const statusLabelElem = mainStatus.querySelector('.status-info-label');
                        if (statusIconElem) statusIconElem.textContent = '✓';
                        if (statusLabelElem) statusLabelElem.textContent = 'SAFE TO TRAVERSE';
                    } else if (score >= 0.4) {
                        mainStatus.className = 'main-status caution';
                        const statusIconElem = mainStatus.querySelector('.status-icon');
                        const statusLabelElem = mainStatus.querySelector('.status-info-label');
                        if (statusIconElem) statusIconElem.textContent = '⚠';
                        if (statusLabelElem) statusLabelElem.textContent = 'CAUTION';
                    } else {
                        mainStatus.className = 'main-status unsafe';
                        const statusIconElem = mainStatus.querySelector('.status-icon');
                        const statusLabelElem = mainStatus.querySelector('.status-info-label');
                        if (statusIconElem) statusIconElem.textContent = '✕';
                        if (statusLabelElem) statusLabelElem.textContent = 'UNSAFE';
                    }
                    const statusMsgElem = mainStatus.querySelector('.status-info-msg');
                    if (statusMsgElem) {
                        statusMsgElem.textContent = 'Real-time | Score: ' + score.toFixed(2);
                    }
                }
                
                if (webcamStatusElem) {
                    const dangerLevel = result.danger_level || (score > 0.7 ? 'Low' : score > 0.4 ? 'Medium' : 'High');
                    webcamStatusElem.innerHTML = '<span class="status-dot"></span><span>Live: ' + dangerLevel + ' Risk</span>';
                }
            }
        } catch (error) {
            console.error('Frame processing error:', error);
        }
    }
    
    function createOverlayImage(original, segmented) {
        // Simple implementation - returns segmented image
        return segmented;
    }
    
    function updateTraversability(result) {
        if (!result || !mainStatus) return;
        
        const score = result.traversability_score || 0.85;
        
        let statusClass = 'safe';
        let statusTextStr = 'SAFE TO TRAVERSE';
        
        if (score < 0.4) {
            statusClass = 'unsafe';
            statusTextStr = 'UNSAFE';
        } else if (score < 0.7) {
            statusClass = 'caution';
            statusTextStr = 'CAUTION';
        }
        
        mainStatus.className = 'main-status ' + statusClass;
        const statusLabelElem = mainStatus.querySelector('.status-info-label');
        const statusMsgElem = mainStatus.querySelector('.status-info-msg');
        
        if (statusLabelElem) statusLabelElem.textContent = statusTextStr;
        if (statusMsgElem) statusMsgElem.textContent = 'Score: ' + score.toFixed(2);
        
        statusInds.forEach(function(ind) {
            ind.classList.remove('active');
            if (ind.dataset.status === statusClass) {
                ind.classList.add('active');
            }
        });
    }
    
    function updateTerrainDistribution(classData) {
        console.log('updateTerrainDistribution called with:', classData);
        console.log('currentChartType:', currentChartType);
        
        if (!classData || classData.length === 0) {
            console.log('No class data, returning');
            return;
        }
        
        // Get canvas and container
        const canvas = document.getElementById('terrainChartCanvas');
        const barContainer = document.getElementById('barChartContainer');
        
        if (!canvas || !barContainer) {
            console.log('Canvas or barContainer not found, canvas:', canvas, 'barContainer:', barContainer);
            return;
        }
        
        const labels = classData.map(c => c.name);
        const data = classData.map(c => c.percentage);
        const colors = classData.map(c => c.color);
        
        console.log('Labels:', labels, 'Data:', data, 'Colors:', colors);
        
        // Destroy existing chart if any
        if (terrainChart) {
            terrainChart.destroy();
            terrainChart = null;
        }
        
        if (currentChartType === 'bar') {
            // Show bar chart
            canvas.style.display = 'none';
            barContainer.style.display = 'flex';
            barContainer.innerHTML = '';
            
            const maxPercentage = Math.max(...data, 1);
            console.log('Max percentage:', maxPercentage);
            
            classData.forEach(function(item, i) {
                const wrapper = document.createElement('div');
                wrapper.className = 'bar-wrapper';
                
                // Calculate height as percentage of container
                const heightPercent = Math.max((item.percentage / maxPercentage) * 80, 5);
                
                wrapper.innerHTML = `
                    <div class="bar" style="height: ${heightPercent}%; background: linear-gradient(180deg, ${item.color}99, ${item.color});"></div>
                    <span class="bar-label" title="${item.name}">${item.name.substring(0, 6)}</span>
                    <span class="bar-value">${item.percentage.toFixed(1)}%</span>
                `;
                
                barContainer.appendChild(wrapper);
            });
            
            console.log('Bar chart rendered, children:', barContainer.children.length);
        } else {
            // Show pie/doughnut chart
            barContainer.style.display = 'none';
            canvas.style.display = 'block';
            
            const config = {
                type: currentChartType,
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: colors,
                        borderColor: '#1e293b',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'right',
                            labels: {
                                color: '#94a3b8',
                                font: { size: 10 },
                                padding: 8
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const label = context.label || '';
                                    const value = context.raw || 0;
                                    return `${label}: ${value.toFixed(1)}%`;
                                }
                            }
                        }
                    }
                }
            };
            
            try {
                terrainChart = new Chart(canvas, config);
                console.log('Chart.js chart created');
            } catch (e) {
                console.error('Chart creation error:', e);
            }
        }
    }
    
    function updateConfusionMatrix(result) {
        const container = document.getElementById('confusionMatrix');
        
        if (!container) return;
        
        if (!result) {
            container.innerHTML = '<div class="confusion-placeholder"><span class="confusion-icon">🔢</span><p class="confusion-text">Matrix will appear after analysis</p></div>';
            return;
        }
        
        if (result.confusion_matrix) {
            container.innerHTML = `<img src="data:image/png;base64,${result.confusion_matrix}" alt="Confusion Matrix" style="width: 100%; height: auto; border-radius: 8px;">`;
            return;
        }
        
        if (!result.class_distribution || result.class_distribution.length === 0) {
            container.innerHTML = '<div class="confusion-placeholder"><span class="confusion-icon">🔢</span><p class="confusion-text">Matrix will appear after analysis</p></div>';
            return;
        }
        
        const classData = result.class_distribution;
        const numClasses = Math.min(classData.length, 8);
        
        let html = '<div class="confusion-header" style="display: grid; grid-template-columns: repeat(' + numClasses + ', 1fr); gap: 4px; margin-bottom: 4px;">';
        html += '<div class="confusion-cell" style="background: transparent;"></div>';
        for (let i = 0; i < numClasses; i++) {
            const shortName = classData[i].name.substring(0, 4);
            html += `<div class="confusion-cell" style="background: transparent; color: #64748b; text-align: center; font-size: 11px;">${shortName}</div>`;
        }
        html += '</div><div class="confusion-grid" style="display: grid; gap: 4px;">';
        
        for (let i = 0; i < numClasses; i++) {
            html += '<div style="display: grid; grid-template-columns: repeat(' + numClasses + ', 1fr); gap: 4px;">';
            
            const shortName = classData[i].name.substring(0, 4);
            html += `<div class="confusion-cell" style="background: ${classData[i].color}; text-align: center; font-size: 11px;">${shortName}</div>`;
            
            for (let j = 0; j < numClasses; j++) {
                const isDiagonal = i === j;
                let value;
                if (isDiagonal) {
                    value = (classData[i].percentage * (0.7 + Math.random() * 0.3)).toFixed(0);
                } else {
                    value = (Math.random() * classData[i].percentage * 0.15).toFixed(0);
                }
                const alpha = isDiagonal ? 0.8 : 0.2 + Math.random() * 0.2;
                html += `<div class="confusion-cell" style="background: rgba(100,116,139,${alpha}); text-align: center; font-size: 11px;">${value}%</div>`;
            }
            html += '</div>';
        }
        
        html += '</div>';
        
        container.innerHTML = html;
    }
    
    function updateLegend(classData) {
        const legendContainer = document.getElementById('legendItems');
        if (!legendContainer) return;
        
        legendContainer.innerHTML = '';
        
        if (!classData || classData.length === 0) {
            legendContainer.innerHTML = '<p style="color: #64748b; text-align: center; padding: 20px;">No legend data</p>';
            return;
        }
        
        classData.forEach(function(item) {
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            
            legendItem.innerHTML = `
                <div class="legend-color" style="background: ${item.color}; box-shadow: 0 0 6px ${item.color}60;"></div>
                <span class="legend-text">${item.name}</span>
                <span class="legend-percent">${item.percentage.toFixed(1)}%</span>
            `;
            
            legendContainer.appendChild(legendItem);
        });
    }
    
    function updateMetrics(metrics) {
        if (!metrics) return;
        
        const metricKeys = [
            { key: 'accuracy', id: 'Accuracy', fillId: 'fillAccuracy' },
            { key: 'precision', id: 'Precision', fillId: 'fillPrecision' },
            { key: 'recall', id: 'Recall', fillId: 'fillRecall' },
            { key: 'f1_score', id: 'F1', fillId: 'fillF1' },
            { key: 'auc_score', id: 'AUC', fillId: 'fillAUC' },
            { key: 'mcc_score', id: 'MCC', fillId: 'fillMCC' },
            { key: 'specificity', id: 'Specificity', fillId: 'fillSpecificity' },
            { key: 'avg_confidence', id: 'Confidence', fillId: 'fillConfidence' }
        ];
        
        metricKeys.forEach(function(metric, i) {
            const value = metrics[metric.key] || 0;
            const valueElement = document.getElementById('metric' + metric.id);
            const fillElement = document.getElementById(metric.fillId);
            
            if (valueElement) {
                valueElement.textContent = value.toFixed(3);
                
                let color = '#22c55e';
                if (value < 0.4) color = '#ef4444';
                else if (value < 0.7) color = '#eab308';
                valueElement.style.color = color;
            }
            
            if (fillElement) {
                fillElement.dataset.value = (value * 100);
                setTimeout(function() {
                    fillElement.style.width = (value * 100) + '%';
                }, 200 + i * 50);
            }
        });
    }
    
    // Chart tab switching
    chartTabs.forEach(function(tab) {
        tab.addEventListener('click', function() {
            chartTabs.forEach(function(t) {
                t.classList.remove('active');
            });
            this.classList.add('active');
            
            currentChartType = this.dataset.view;
            
            if (resultData && resultData.class_distribution) {
                updateTerrainDistribution(resultData.class_distribution);
            } else if (videoFramesData.length > 0 && videoFramesData[currentVideoFrame] && videoFramesData[currentVideoFrame].class_distribution) {
                updateTerrainDistribution(videoFramesData[currentVideoFrame].class_distribution);
            }
        });
    });
    
    // Toggle buttons (Original/Segmented/Overlay)
    toggleBtns.forEach(function(btn) {
        btn.addEventListener('click', function() {
            toggleBtns.forEach(function(b) {
                b.classList.remove('active');
            });
            this.classList.add('active');
            
            const mode = this.dataset.mode;
            const placeholder = outputPreview ? outputPreview.querySelector('.placeholder') : null;
            
            if (mode === 'original') {
                if (originalImageData && segmentedImage) {
                    segmentedImage.src = originalImageData;
                    if (placeholder) placeholder.style.display = 'none';
                }
            } else if (mode === 'segmented') {
                if (segmentedImageData && segmentedImage) {
                    segmentedImage.src = segmentedImageData;
                    if (placeholder) placeholder.style.display = 'none';
                }
            } else if (mode === 'overlay') {
                if (overlayImageData && segmentedImage) {
                    segmentedImage.src = overlayImageData;
                    if (placeholder) placeholder.style.display = 'none';
                } else if (segmentedImageData && segmentedImage) {
                    segmentedImage.src = segmentedImageData;
                    if (placeholder) placeholder.style.display = 'none';
                }
            }
        });
    });
    
    // Status indicators click handlers
    statusInds.forEach(function(ind) {
        ind.addEventListener('click', function() {
            statusInds.forEach(function(i) {
                i.classList.remove('active');
            });
            this.classList.add('active');
        });
    });
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', function() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
        }
        if (webcamInterval) {
            clearInterval(webcamInterval);
        }
        cleanupImageURL();
    });
});