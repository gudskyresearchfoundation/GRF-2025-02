/**
 * AI-Driven Precision Agriculture - Frontend JavaScript
 * Project ID: GRF-2025-02
 * Company: Gudsky Research Foundation
 */

// API Base URL
const API_BASE = '/api';

// Global state to store analysis results
const analysisResults = {};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    initializeTabs();
    initializeImageUpload();
    initializeNavigation();
    console.log('🌱 Precision Agriculture System Initialized');
});

// ============================================================================
// TAB NAVIGATION
// ============================================================================

function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.getAttribute('data-tab');
            
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');
        });
    });
}

// ============================================================================
// SMOOTH NAVIGATION
// ============================================================================

function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href.startsWith('#')) {
                e.preventDefault();
                const targetId = href.substring(1);
                scrollToSection(targetId);
                
                // Update active nav link
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            }
        });
    });
}

function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// ============================================================================
// IMAGE UPLOAD - DISEASE DETECTION
// ============================================================================

function initializeImageUpload() {
    const uploadArea = document.getElementById('diseaseUploadArea');
    const fileInput = document.getElementById('diseaseImageInput');
    const analyzeBtn = document.getElementById('analyzeDiseaseBtn');
    
    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // File selection
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleImageFile(file);
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImageFile(file);
        } else {
            showNotification('Please upload an image file', 'error');
        }
    });
}

function handleImageFile(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        const preview = document.getElementById('diseasePreview');
        const previewImg = document.getElementById('diseasePreviewImg');
        const uploadArea = document.getElementById('diseaseUploadArea');
        const analyzeBtn = document.getElementById('analyzeDiseaseBtn');
        
        previewImg.src = e.target.result;
        preview.style.display = 'block';
        uploadArea.style.display = 'none';
        analyzeBtn.disabled = false;
        
        // Store file for analysis
        window.diseaseImageFile = file;
    };
    
    reader.readAsDataURL(file);
}

function removeDiseaseImage() {
    const preview = document.getElementById('diseasePreview');
    const uploadArea = document.getElementById('diseaseUploadArea');
    const fileInput = document.getElementById('diseaseImageInput');
    const analyzeBtn = document.getElementById('analyzeDiseaseBtn');
    const results = document.getElementById('diseaseResults');
    
    preview.style.display = 'none';
    uploadArea.style.display = 'flex';
    fileInput.value = '';
    analyzeBtn.disabled = true;
    results.style.display = 'none';
    window.diseaseImageFile = null;
}

// ============================================================================
// DISEASE DETECTION ANALYSIS
// ============================================================================

document.getElementById('analyzeDiseaseBtn').addEventListener('click', async function() {
    if (!window.diseaseImageFile) {
        showNotification('Please upload an image first', 'error');
        return;
    }
    
    showLoading();
    
    const formData = new FormData();
    formData.append('file', window.diseaseImageFile);
    
    try {
        const response = await fetch(`${API_BASE}/predict/disease`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Disease detection failed');
        }
        
        const result = await response.json();
        analysisResults.disease = result;
        displayDiseaseResults(result);
        showNotification('Disease analysis complete!', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error analyzing image. Please try again.', 'error');
    } finally {
        hideLoading();
    }
});

function displayDiseaseResults(result) {
    const resultsDiv = document.getElementById('diseaseResults');
    
    const statusClass = result.health_status === 'HEALTHY' ? 'status-healthy' : 'status-unhealthy';
    const statusIcon = result.health_status === 'HEALTHY' ? '✅' : '⚠️';
    
    const html = `
        <div class="result-header">
            <h4>Disease Detection Results</h4>
            <span class="result-status ${statusClass}">${statusIcon} ${result.health_status}</span>
        </div>
        
        <div class="result-grid">
            <div class="result-item">
                <div class="result-label">Crop Detected</div>
                <div class="result-value">${result.crop}</div>
            </div>
            <div class="result-item">
                <div class="result-label">Disease/Condition</div>
                <div class="result-value">${result.disease}</div>
            </div>
            <div class="result-item">
                <div class="result-label">Confidence</div>
                <div class="result-value">${result.confidence}%</div>
            </div>
            <div class="result-item">
                <div class="result-label">Model Used</div>
                <div class="result-value">ResNet50</div>
            </div>
        </div>
        
        ${result.warning ? `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle"></i> ${result.warning}
            </div>
        ` : ''}
        
        <div class="recommendations-section">
            <h4>Recommendations</h4>
            <ul class="recommendations-list">
                ${result.health_status === 'UNHEALTHY' ? `
                    <li>🔴 Disease detected: ${result.disease}</li>
                    <li>📋 Consult agricultural expert for treatment plan</li>
                    <li>🔬 Consider laboratory diagnosis for confirmation</li>
                    <li>🚿 Isolate affected plants to prevent spread</li>
                    <li>💊 Apply appropriate fungicide/pesticide as recommended</li>
                ` : `
                    <li>✅ Crop appears healthy</li>
                    <li>👀 Continue regular monitoring</li>
                    <li>🌱 Maintain current care practices</li>
                `}
            </ul>
        </div>
    `;
    
    resultsDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
    
    // Track analysis for Balarama Ji chat
    updateChatContext('disease', result);
}

// ============================================================================
// SOIL ANALYSIS
// ============================================================================

async function analyzeSoil() {
    const ph = parseFloat(document.getElementById('soilPH').value);
    const nitrogen = parseFloat(document.getElementById('soilNitrogen').value);
    const phosphorus = parseFloat(document.getElementById('soilPhosphorus').value);
    const potassium = parseFloat(document.getElementById('soilPotassium').value);
    const organic = document.getElementById('soilOrganic').value;
    const moisture = document.getElementById('soilMoisture').value;
    
    // Validation
    if (!ph || !nitrogen || !phosphorus || !potassium) {
        showNotification('Please fill in all required fields', 'error');
        return;
    }
    
    showLoading();
    
    const data = {
        ph,
        nitrogen,
        phosphorus,
        potassium
    };
    
    if (organic) data.organic_carbon = parseFloat(organic);
    if (moisture) data.moisture = parseFloat(moisture);
    
    try {
        const response = await fetch(`${API_BASE}/soil/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) throw new Error('Soil analysis failed');
        
        const result = await response.json();
        analysisResults.soil = result;
        displaySoilResults(result);
        showNotification('Soil analysis complete!', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error analyzing soil data', 'error');
    } finally {
        hideLoading();
    }
}

function displaySoilResults(result) {
    const resultsDiv = document.getElementById('soilResults');
    
    let statusClass = 'status-healthy';
    if (result.health_score < 40) statusClass = 'status-unhealthy';
    else if (result.health_score < 70) statusClass = 'status-warning';
    
    const html = `
        <div class="result-header">
            <h4>Soil Health Analysis</h4>
            <span class="result-status ${statusClass}">
                ${result.health_status} (Score: ${result.health_score}/100)
            </span>
        </div>
        
        <div class="result-grid">
            <div class="result-item">
                <div class="result-label">pH Score</div>
                <div class="result-value">${result.parameter_scores.ph}/100</div>
            </div>
            <div class="result-item">
                <div class="result-label">Nitrogen Score</div>
                <div class="result-value">${result.parameter_scores.nitrogen}/100</div>
            </div>
            <div class="result-item">
                <div class="result-label">Phosphorus Score</div>
                <div class="result-value">${result.parameter_scores.phosphorus}/100</div>
            </div>
            <div class="result-item">
                <div class="result-label">Potassium Score</div>
                <div class="result-value">${result.parameter_scores.potassium}/100</div>
            </div>
        </div>
        
        <div class="recommendations-section">
            <h4>NPK Fertilizer Recommendations</h4>
            <div class="result-grid">
                <div class="result-item">
                    <div class="result-label">Urea (N)</div>
                    <div class="result-value">${result.npk_fertilizer.recommended_fertilizers.urea_kg_ha} kg/ha</div>
                </div>
                <div class="result-item">
                    <div class="result-label">DAP (P)</div>
                    <div class="result-value">${result.npk_fertilizer.recommended_fertilizers.dap_kg_ha} kg/ha</div>
                </div>
                <div class="result-item">
                    <div class="result-label">MOP (K)</div>
                    <div class="result-value">${result.npk_fertilizer.recommended_fertilizers.mop_kg_ha} kg/ha</div>
                </div>
            </div>
            <p><strong>Application Timing:</strong> ${result.npk_fertilizer.application_timing}</p>
        </div>
        
        <div class="recommendations-section">
            <h4>Recommendations</h4>
            <ul class="recommendations-list">
                ${result.recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
        
        <div class="summary-box">
            <p><strong>Summary:</strong> ${result.summary}</p>
        </div>
    `;
    
    resultsDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
    
    // Track analysis for Balarama Ji chat
    updateChatContext('soil', result);
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Continue to app_part2.js...
// ============================================================================
// WEATHER FORECAST
// ============================================================================

async function getWeather() {
    const lat = parseFloat(document.getElementById('weatherLat').value);
    const lon = parseFloat(document.getElementById('weatherLon').value);
    const days = parseInt(document.getElementById('weatherDays').value);
    
    if (!lat || !lon) {
        showNotification('Please enter valid coordinates', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/weather?latitude=${lat}&longitude=${lon}&days=${days}`);
        
        if (!response.ok) throw new Error('Weather fetch failed');
        
        const result = await response.json();
        analysisResults.weather = result;
        displayWeatherResults(result);
        showNotification('Weather data loaded!', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error fetching weather data', 'error');
    } finally {
        hideLoading();
    }
}

async function searchLocation() {
    const query = document.getElementById('locationSearch').value;
    
    if (!query) {
        showNotification('Please enter a location name', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/weather/search-location?query=${encodeURIComponent(query)}`);
        
        if (!response.ok) throw new Error('Location search failed');
        
        const result = await response.json();
        
        if (result.locations && result.locations.length > 0) {
            const location = result.locations[0];
            document.getElementById('weatherLat').value = location.latitude;
            document.getElementById('weatherLon').value = location.longitude;
            showNotification(`Found: ${location.name}, ${location.country}`, 'success');
        } else {
            showNotification('Location not found', 'error');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error searching location', 'error');
    } finally {
        hideLoading();
    }
}

function displayWeatherResults(result) {
    const resultsDiv = document.getElementById('weatherResults');
    const current = result.weather.current_weather || {};
    const alerts = result.agricultural_alerts || [];
    
    const html = `
        <div class="result-header">
            <h4>Weather Forecast</h4>
            <span class="result-status status-healthy">
                <i class="fas fa-map-marker-alt"></i> 
                Lat: ${result.location.latitude.toFixed(4)}, Lon: ${result.location.longitude.toFixed(4)}
            </span>
        </div>
        
        <div class="result-grid">
            <div class="result-item">
                <div class="result-label">Current Temperature</div>
                <div class="result-value">${current.temperature || 'N/A'}°C</div>
            </div>
            <div class="result-item">
                <div class="result-label">Wind Speed</div>
                <div class="result-value">${current.windspeed || 'N/A'} km/h</div>
            </div>
            <div class="result-item">
                <div class="result-label">Forecast Period</div>
                <div class="result-value">${result.forecast_days} days</div>
            </div>
            <div class="result-item">
                <div class="result-label">Conditions</div>
                <div class="result-value">${getWeatherIcon(current.weathercode)}</div>
            </div>
        </div>
        
        <div class="recommendations-section">
            <h4>Agricultural Alerts</h4>
            <ul class="recommendations-list">
                ${alerts.map(alert => `<li>${alert}</li>`).join('')}
            </ul>
        </div>
    `;
    
    resultsDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
    
    // Track analysis for Balarama Ji chat
    updateChatContext('weather', result);
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function getWeatherIcon(code) {
    const icons = {
        0: '☀️ Clear',
        1: '🌤️ Mainly Clear',
        2: '⛅ Partly Cloudy',
        3: '☁️ Overcast',
        45: '🌫️ Foggy',
        61: '🌧️ Light Rain',
        63: '🌧️ Moderate Rain',
        65: '🌧️ Heavy Rain',
        95: '⛈️ Thunderstorm'
    };
    return icons[code] || '🌤️ Fair';
}

// ============================================================================
// YIELD PREDICTION
// ============================================================================

function addHistoricalEntry() {
    const container = document.getElementById('historicalDataInputs');
    const entry = document.createElement('div');
    entry.className = 'historical-entry';
    entry.innerHTML = `
        <input type="date" placeholder="Date">
        <input type="number" placeholder="Yield (kg/ha)" step="1">
        <button class="btn btn-remove-entry" onclick="removeHistoricalEntry(this)">
            <i class="fas fa-minus"></i>
        </button>
    `;
    container.appendChild(entry);
}

function removeHistoricalEntry(button) {
    const entries = document.querySelectorAll('.historical-entry');
    if (entries.length > 1) {
        button.parentElement.remove();
    }
}

async function predictYield() {
    const crop = document.getElementById('yieldCrop').value;
    const forecastDays = parseInt(document.getElementById('forecastDays').value);
    
    // Collect historical data
    const entries = document.querySelectorAll('.historical-entry');
    const historicalYields = [];
    
    entries.forEach(entry => {
        const date = entry.querySelector('input[type="date"]').value;
        const yieldValue = entry.querySelector('input[type="number"]').value;
        
        if (date && yieldValue) {
            historicalYields.push({
                date: date,
                yield: parseFloat(yieldValue)
            });
        }
    });
    
    showLoading();
    
    const data = {
        historical_yields: historicalYields,
        crop_type: crop,
        forecast_days: forecastDays
    };
    
    try {
        const response = await fetch(`${API_BASE}/yield/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) throw new Error('Yield prediction failed');
        
        const result = await response.json();
        analysisResults.yield = result;
        displayYieldResults(result);
        showNotification('Yield prediction complete!', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error predicting yield', 'error');
    } finally {
        hideLoading();
    }
}

function displayYieldResults(result) {
    const resultsDiv = document.getElementById('yieldResults');
    
    const trendIcon = result.trend === 'increasing' ? '📈' : result.trend === 'decreasing' ? '📉' : '➡️';
    
    const html = `
        <div class="result-header">
            <h4>Yield Prediction Results</h4>
            <span class="result-status status-healthy">
                ${trendIcon} ${result.trend.toUpperCase()}
            </span>
        </div>
        
        <div class="result-grid">
            <div class="result-item">
                <div class="result-label">Crop Type</div>
                <div class="result-value">${result.crop_type}</div>
            </div>
            <div class="result-item">
                <div class="result-label">Predicted Yield</div>
                <div class="result-value">${result.predicted_average_yield} kg/ha</div>
            </div>
            <div class="result-item">
                <div class="result-label">Growth Rate</div>
                <div class="result-value">${result.growth_rate_percent}%</div>
            </div>
            <div class="result-item">
                <div class="result-label">Method</div>
                <div class="result-value">${result.prediction_method}</div>
            </div>
        </div>
        
        <div class="yield-range">
            <h4>Predicted Range</h4>
            <p>Lower Bound: <strong>${result.predicted_yield_range.lower} kg/ha</strong></p>
            <p>Upper Bound: <strong>${result.predicted_yield_range.upper} kg/ha</strong></p>
            <p>Confidence: ${result.confidence_interval}</p>
        </div>
        
        <div class="recommendations-section">
            <h4>Recommendations</h4>
            <ul class="recommendations-list">
                ${result.recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
        
        ${result.note ? `
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i> ${result.note}
            </div>
        ` : ''}
    `;
    
    resultsDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
    
    // Track analysis for Balarama Ji chat
    updateChatContext('yield', result);
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ============================================================================
// IRRIGATION CALCULATION
// ============================================================================

async function calculateIrrigation() {
    const crop = document.getElementById('irrigCrop').value;
    const moisture = parseFloat(document.getElementById('irrigMoisture').value);
    const stage = document.getElementById('irrigStage').value;
    const rainfall = parseFloat(document.getElementById('irrigRainfall').value);
    const temp = parseFloat(document.getElementById('irrigTemp').value);
    
    if (isNaN(moisture) || isNaN(rainfall) || isNaN(temp)) {
        showNotification('Please fill in all fields', 'error');
        return;
    }
    
    showLoading();
    
    const data = {
        crop_type: crop,
        soil_moisture: moisture,
        growth_stage: stage,
        rainfall_forecast_mm: rainfall,
        temperature_celsius: temp
    };
    
    try {
        const response = await fetch(`${API_BASE}/irrigation/calculate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) throw new Error('Irrigation calculation failed');
        
        const result = await response.json();
        analysisResults.irrigation = result;
        displayIrrigationResults(result);
        showNotification('Irrigation schedule calculated!', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error calculating irrigation', 'error');
    } finally {
        hideLoading();
    }
}

function displayIrrigationResults(result) {
    const resultsDiv = document.getElementById('irrigationResults');
    
    let statusClass = 'status-healthy';
    if (result.soil_moisture_percent < 30) statusClass = 'status-unhealthy';
    else if (result.soil_moisture_percent < 50) statusClass = 'status-warning';
    
    const html = `
        <div class="result-header">
            <h4>Irrigation Schedule</h4>
            <span class="result-status ${statusClass}">
                💧 ${result.soil_moisture_percent}% Moisture
            </span>
        </div>
        
        <div class="result-grid">
            <div class="result-item">
                <div class="result-label">Crop</div>
                <div class="result-value">${result.crop_type}</div>
            </div>
            <div class="result-item">
                <div class="result-label">Growth Stage</div>
                <div class="result-value">${result.growth_stage}</div>
            </div>
            <div class="result-item">
                <div class="result-label">Water Needed</div>
                <div class="result-value">${result.irrigation_amount_mm_per_day} mm/day</div>
            </div>
            <div class="result-item">
                <div class="result-label">Next Irrigation</div>
                <div class="result-value">${result.next_irrigation_days} days</div>
            </div>
        </div>
        
        <div class="recommendations-section">
            <h4>Irrigation Recommendations</h4>
            <ul class="recommendations-list">
                ${result.recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
    `;
    
    resultsDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
    
    // Track analysis for Balarama Ji chat
    updateChatContext('irrigation', result);
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ============================================================================
// PEST RISK ASSESSMENT
// ============================================================================

async function assessPestRisk() {
    const crop = document.getElementById('pestCrop').value;
    const temp = parseFloat(document.getElementById('pestTemp').value);
    const humidity = parseFloat(document.getElementById('pestHumidity').value);
    const rainfall = parseFloat(document.getElementById('pestRainfall').value);
    const previousInfestation = document.getElementById('pestPreviousInfestation').checked;
    
    if (isNaN(temp) || isNaN(humidity) || isNaN(rainfall)) {
        showNotification('Please fill in all fields', 'error');
        return;
    }
    
    showLoading();
    
    const data = {
        crop_type: crop,
        temperature: temp,
        humidity: humidity,
        rainfall_7days: rainfall,
        previous_infestation: previousInfestation
    };
    
    try {
        const response = await fetch(`${API_BASE}/pest/assess`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) throw new Error('Pest assessment failed');
        
        const result = await response.json();
        analysisResults.pest = result;
        displayPestResults(result);
        showNotification('Pest risk assessment complete!', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error assessing pest risk', 'error');
    } finally {
        hideLoading();
    }
}

function displayPestResults(result) {
    const resultsDiv = document.getElementById('pestResults');
    
    let statusClass = 'status-healthy';
    if (result.risk_level === 'High') statusClass = 'status-unhealthy';
    else if (result.risk_level === 'Medium') statusClass = 'status-warning';
    
    const html = `
        <div class="result-header">
            <h4>Pest Risk Assessment</h4>
            <span class="result-status ${statusClass}">
                ${result.risk_indicator} ${result.risk_level} Risk
            </span>
        </div>
        
        <div class="result-grid">
            <div class="result-item">
                <div class="result-label">Crop</div>
                <div class="result-value">${result.crop_type}</div>
            </div>
            <div class="result-item">
                <div class="result-label">Risk Score</div>
                <div class="result-value">${result.risk_score}/100</div>
            </div>
            <div class="result-item">
                <div class="result-label">Monitoring Frequency</div>
                <div class="result-value">${result.monitoring_frequency}</div>
            </div>
        </div>
        
        <div class="contributing-factors">
            <h4>Contributing Factors</h4>
            <ul class="recommendations-list">
                ${result.contributing_factors.map(factor => `<li>${factor}</li>`).join('')}
            </ul>
        </div>
        
        <div class="likely-pests">
            <h4>Likely Pests for ${result.crop_type}</h4>
            <ul class="recommendations-list">
                ${result.likely_pests.map(pest => `<li>🐛 ${pest}</li>`).join('')}
            </ul>
        </div>
        
        <div class="recommendations-section">
            <h4>Preventive Measures</h4>
            <ul class="recommendations-list">
                ${result.preventive_measures.map(measure => `<li>${measure}</li>`).join('')}
            </ul>
        </div>
    `;
    
    resultsDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
    
    // Track analysis for Balarama Ji chat
    updateChatContext('pest', result);
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Continue to app_part3.js for report generation and utilities...
// ============================================================================
// COMPREHENSIVE REPORT GENERATION
// ============================================================================

async function generateReport() {
    const hasData = Object.keys(analysisResults).length > 0;
    
    if (!hasData) {
        showNotification('Please complete at least one analysis before generating a report', 'warning');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/report/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(analysisResults)
        });
        
        if (!response.ok) throw new Error('Report generation failed');
        
        const result = await response.json();
        displayReport(result.report);
        showNotification('Report generated successfully!', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error generating report', 'error');
    } finally {
        hideLoading();
    }
}

function displayReport(reportText) {
    const reportOutput = document.getElementById('reportOutput');
    reportOutput.textContent = reportText;
    reportOutput.style.display = 'block';
    
    // Scroll to report
    document.getElementById('reports').scrollIntoView({ behavior: 'smooth' });
    
    // Add download button
    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'btn btn-secondary mt-2';
    downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download Report';
    downloadBtn.onclick = () => downloadReport(reportText);
    
    // Add copy button
    const copyBtn = document.createElement('button');
    copyBtn.className = 'btn btn-secondary mt-2';
    copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy to Clipboard';
    copyBtn.onclick = () => copyToClipboard(reportText);
    
    const buttonContainer = document.createElement('div');
    buttonContainer.style.marginTop = '16px';
    buttonContainer.style.display = 'flex';
    buttonContainer.style.gap = '16px';
    buttonContainer.appendChild(downloadBtn);
    buttonContainer.appendChild(copyBtn);
    
    reportOutput.parentNode.insertBefore(buttonContainer, reportOutput.nextSibling);
}

function downloadReport(reportText) {
    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Agriculture_Report_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showNotification('Report downloaded!', 'success');
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Report copied to clipboard!', 'success');
    }).catch(() => {
        showNotification('Failed to copy report', 'error');
    });
}

// ============================================================================
// LOADING SPINNER
// ============================================================================

function showLoading() {
    document.getElementById('loadingSpinner').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingSpinner').style.display = 'none';
}

// ============================================================================
// NOTIFICATIONS
// ============================================================================

function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    const icon = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
    }[type] || 'ℹ️';
    
    notification.innerHTML = `
        <span>${icon} ${message}</span>
        <button onclick="this.parentElement.remove()">×</button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideOut 0.3s ease-out forwards';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

// Add notification styles dynamically
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    .notification {
        position: fixed;
        top: 80px;
        right: 20px;
        padding: 16px 24px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        gap: 12px;
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
        font-weight: 500;
        max-width: 400px;
    }
    
    .notification-success {
        background: #27ae60;
        color: white;
    }
    
    .notification-error {
        background: #e74c3c;
        color: white;
    }
    
    .notification-warning {
        background: #f39c12;
        color: white;
    }
    
    .notification-info {
        background: #3498db;
        color: white;
    }
    
    .notification button {
        background: none;
        border: none;
        color: white;
        font-size: 24px;
        cursor: pointer;
        padding: 0;
        margin-left: auto;
        line-height: 1;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
    
    .alert {
        padding: 12px 16px;
        border-radius: 8px;
        margin: 16px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .alert-warning {
        background: rgba(243, 156, 18, 0.1);
        border-left: 4px solid #f39c12;
        color: #e67e22;
    }
    
    .alert-info {
        background: rgba(52, 152, 219, 0.1);
        border-left: 4px solid #3498db;
        color: #2980b9;
    }
    
    .yield-range {
        background: white;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }
    
    .yield-range h4 {
        color: #2ecc71;
        margin-bottom: 12px;
    }
    
    .yield-range p {
        margin: 8px 0;
        color: #2c3e50;
    }
    
    .summary-box {
        background: #ecf0f1;
        padding: 16px;
        border-radius: 8px;
        margin-top: 16px;
        border-left: 4px solid #2ecc71;
    }
    
    .summary-box p {
        margin: 0;
        color: #2c3e50;
    }
    
    .contributing-factors,
    .likely-pests {
        margin: 16px 0;
    }
    
    .contributing-factors h4,
    .likely-pests h4 {
        color: #2ecc71;
        margin-bottom: 12px;
    }
    
    .recommendations-section {
        margin: 24px 0;
    }
    
    .recommendations-section h4 {
        color: #2ecc71;
        margin-bottom: 16px;
    }
`;
document.head.appendChild(notificationStyles);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Format numbers
function formatNumber(num, decimals = 2) {
    return Number(num).toFixed(decimals);
}

// Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
    });
}

// Check if API is available
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        console.log('API Health:', data);
        return data.status === 'healthy';
    } catch (error) {
        console.error('API Health Check Failed:', error);
        return false;
    }
}

// Initialize API health check on load
window.addEventListener('load', async () => {
    const isHealthy = await checkAPIHealth();
    if (!isHealthy) {
        showNotification('Warning: API may be unavailable. Some features may not work.', 'warning');
    }
});

// ============================================================================
// KEYBOARD SHORTCUTS
// ============================================================================

document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        document.getElementById('locationSearch')?.focus();
    }
    
    // Escape to close modals/clear results
    if (e.key === 'Escape') {
        hideLoading();
    }
});

// ============================================================================
// SCROLL ANIMATIONS
// ============================================================================

const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all cards for animation
document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.analysis-card, .report-card, .feature-item');
    cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });
});

// ============================================================================
// FORM VALIDATION
// ============================================================================

function validateNumber(value, min, max, fieldName) {
    const num = parseFloat(value);
    
    if (isNaN(num)) {
        showNotification(`${fieldName} must be a valid number`, 'error');
        return false;
    }
    
    if (num < min || num > max) {
        showNotification(`${fieldName} must be between ${min} and ${max}`, 'error');
        return false;
    }
    
    return true;
}

// ============================================================================
// DATA EXPORT
// ============================================================================

function exportAnalysisResults() {
    const dataStr = JSON.stringify(analysisResults, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis_results_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showNotification('Analysis results exported!', 'success');
}

// Add export button if results exist
setInterval(() => {
    const generateBtn = document.getElementById('generateReportBtn');
    if (generateBtn && Object.keys(analysisResults).length > 0) {
        if (!document.getElementById('exportDataBtn')) {
            const exportBtn = document.createElement('button');
            exportBtn.id = 'exportDataBtn';
            exportBtn.className = 'btn btn-secondary';
            exportBtn.innerHTML = '<i class="fas fa-file-export"></i> Export Data (JSON)';
            exportBtn.onclick = exportAnalysisResults;
            generateBtn.parentNode.appendChild(exportBtn);
        }
    }
}, 2000);

// ============================================================================
// PRINT FUNCTIONALITY
// ============================================================================

function printReport() {
    window.print();
}

// Add print styles
const printStyles = document.createElement('style');
printStyles.textContent = `
    @media print {
        nav, .tab-navigation, .hero, .footer, button {
            display: none !important;
        }
        
        .results-container {
            page-break-inside: avoid;
        }
        
        body {
            background: white;
        }
    }
`;
document.head.appendChild(printStyles);

// ============================================================================
// CONSOLE WELCOME MESSAGE
// ============================================================================

console.log('%c🌱 Precision Agriculture System', 'color: #2ecc71; font-size: 20px; font-weight: bold;');
console.log('%cProject ID: GRF-2025-02', 'color: #3498db; font-size: 14px;');
console.log('%cGudsky Research Foundation', 'color: #2c3e50; font-size: 14px;');
console.log('%c━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'color: #95a5a6;');
console.log('API Documentation: http://localhost:8000/docs');
console.log('System Status: Operational');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

// ============================================================================
// ERROR BOUNDARY
// ============================================================================

window.addEventListener('error', (event) => {
    console.error('Global Error:', event.error);
    hideLoading();
    showNotification('An unexpected error occurred. Please refresh and try again.', 'error');
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled Promise Rejection:', event.reason);
    hideLoading();
    showNotification('An unexpected error occurred. Please try again.', 'error');
});

// ============================================================================
// SERVICE WORKER (Optional - for PWA)
// ============================================================================

if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Uncomment to enable PWA
        // navigator.serviceWorker.register('/sw.js')
        //     .then(reg => console.log('Service Worker registered:', reg))
        //     .catch(err => console.log('Service Worker registration failed:', err));
    });
}

// ============================================================================
// ANALYTICS (Optional)
// ============================================================================

function trackEvent(category, action, label) {
    console.log(`Event: ${category} - ${action} - ${label}`);
    // Add analytics tracking here (Google Analytics, etc.)
}

// Track tab switches
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        trackEvent('Navigation', 'Tab Switch', btn.getAttribute('data-tab'));
    });
});

// ============================================================================
// END OF APP.JS
// ============================================================================

console.log('✅ All modules loaded successfully');


// ============================================================================
// BALARAMA JI CHAT ASSISTANT - Divine Farm Guidance
// ============================================================================

/**
// ============================================================================
// BALARAMAJI CHAT ASSISTANT - Divine Agricultural Guidance
// ============================================================================

// Chat history storage
let chatHistory = [];

// ============================================================================
// CHAT INITIALIZATION
// ============================================================================

function initializeChat() {
    console.log('🙏 Balaramaji Chat Assistant initialized');
    
    updateAnalysisSummaryPanel();
    
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', handleChatKeyPress);
    }
    
    const sendBtn = document.getElementById('chatSendBtn');
    if (sendBtn) {
        sendBtn.addEventListener('click', sendChatMessage);
    }
    
    loadChatHistory();
}

setTimeout(initializeChat, 100);

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

function formatAnalysisContext() {
    const context = {};
    
    if (analysisResults.disease) {
        context.disease = { data: analysisResults.disease };
    }
    if (analysisResults.soil) {
        context.soil = { data: analysisResults.soil };
    }
    if (analysisResults.weather) {
        context.weather = { data: analysisResults.weather };
    }
    if (analysisResults.yield) {
        context.yield = { data: analysisResults.yield };
    }
    if (analysisResults.irrigation) {
        context.irrigation = { data: analysisResults.irrigation };
    }
    if (analysisResults.pest) {
        context.pest = { data: analysisResults.pest };
    }
    
    return Object.keys(context).length > 0 ? context : null;
}

function getChatHistory() {
    return chatHistory.slice(-8);
}

function updateChatHistory(userMsg, assistantMsg) {
    chatHistory.push(
        { role: 'user', content: userMsg },
        { role: 'assistant', content: assistantMsg }
    );
    
    if (chatHistory.length > 50) {
        chatHistory = chatHistory.slice(-50);
    }
    
    saveChatHistory();
}

function saveChatHistory() {
    try {
        localStorage.setItem('balaramaji_chat_history', JSON.stringify(chatHistory));
    } catch (e) {
        console.warn('Could not save chat history:', e);
    }
}

function loadChatHistory() {
    try {
        const saved = localStorage.getItem('balaramaji_chat_history');
        if (saved) {
            chatHistory = JSON.parse(saved);
            console.log('✅ Chat history loaded');
        }
    } catch (e) {
        console.warn('Could not load chat history:', e);
        chatHistory = [];
    }
}

function clearChatHistory() {
    chatHistory = [];
    localStorage.removeItem('balaramaji_chat_history');
    
    const messagesContainer = document.getElementById('chatMessages');
    if (messagesContainer) {
        messagesContainer.innerHTML = `
            <div class="message assistant-message">
                <div class="message-avatar">🌱</div>
                <div class="message-content">
                    <p>🙏 Jai Shri Balarama! Blessed farmer, I am here to guide you with divine wisdom.</p>
                    <p>Complete your farm analyses and ask me any questions about your crops, soil, irrigation, or farming practices!</p>
                </div>
            </div>
        `;
    }
    
    showNotification('Chat history cleared', 'success');
}

// ============================================================================
// ANALYSIS SUMMARY PANEL
// ============================================================================

function updateAnalysisSummaryPanel() {
    const summaryContent = document.getElementById('analysisSummaryContent');
    if (!summaryContent) return;
    
    const analyses = [];
    
    if (analysisResults.disease) {
        const disease = analysisResults.disease;
        analyses.push(`
            <div class="summary-item">
                <i class="fas fa-microscope"></i>
                <strong>Disease:</strong> 
                ${disease.crop} - ${disease.disease} 
                (${disease.confidence}% confident)
            </div>
        `);
    }
    
    if (analysisResults.soil) {
        const soil = analysisResults.soil;
        analyses.push(`
            <div class="summary-item">
                <i class="fas fa-seedling"></i>
                <strong>Soil:</strong> 
                pH ${soil.ph}, Health ${soil.health_score}/100
            </div>
        `);
    }
    
    if (analysisResults.weather) {
        const weather = analysisResults.weather;
        const temp = weather.weather?.current_weather?.temperature || 'N/A';
        analyses.push(`
            <div class="summary-item">
                <i class="fas fa-cloud-sun"></i>
                <strong>Weather:</strong> 
                ${temp}°C
            </div>
        `);
    }
    
    if (analysisResults.yield) {
        const yieldData = analysisResults.yield;
        analyses.push(`
            <div class="summary-item">
                <i class="fas fa-chart-line"></i>
                <strong>Yield:</strong> 
                ${yieldData.predicted_average_yield} kg/ha
            </div>
        `);
    }
    
    if (analysisResults.irrigation) {
        const irrigation = analysisResults.irrigation;
        analyses.push(`
            <div class="summary-item">
                <i class="fas fa-tint"></i>
                <strong>Irrigation:</strong> 
                ${irrigation.soil_moisture_percent}% moisture, 
                ${irrigation.irrigation_amount_mm_per_day}mm/day needed
            </div>
        `);
    }
    
    if (analysisResults.pest) {
        const pest = analysisResults.pest;
        analyses.push(`
            <div class="summary-item">
                <i class="fas fa-bug"></i>
                <strong>Pest Risk:</strong> 
                ${pest.risk_level} (${pest.risk_score}/100)
            </div>
        `);
    }
    
    if (analyses.length > 0) {
        summaryContent.innerHTML = analyses.join('');
    } else {
        summaryContent.innerHTML = '<p class="no-data">No analyses completed yet. Complete an analysis to get personalized recommendations!</p>';
    }
}

function updateChatContext(analysisType, result) {
    analysisResults[analysisType] = result;
    updateAnalysisSummaryPanel();
}

// ============================================================================
// CHAT MESSAGING
// ============================================================================

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    displayChatMessage(message, 'user');
    input.value = '';
    document.getElementById('typingIndicator').style.display = 'block';
    
    const requestData = {
        message: message,
        analysis_context: formatAnalysisContext(),
        chat_history: getChatHistory()
    };
    
    try {
        const response = await fetch(`${API_BASE}/chat/balaramaji`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        document.getElementById('typingIndicator').style.display = 'none';
        displayChatMessage(result.response, 'assistant');
        updateChatHistory(message, result.response);
        updateAnalysisSummaryPanel();
        
        console.log('✅ Balaramaji response:', {
            source: result.source,
            model: result.model,
            has_context: result.has_context
        });
        
    } catch (error) {
        console.error('❌ Chat error:', error);
        document.getElementById('typingIndicator').style.display = 'none';
        displayChatMessage(
            '🙏 Forgive me, blessed farmer. I am experiencing difficulties connecting to the divine wisdom. Please ensure Ollama is running and try again.',
            'assistant'
        );
        showNotification('Chat error: ' + error.message, 'error');
    }
}

function displayChatMessage(message, role) {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    const avatar = role === 'user' ? '👨‍🌾' : '🌱';
    const time = new Date().toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            ${formatMessageContent(message)}
            <div class="message-time">${time}</div>
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function formatMessageContent(text) {
    text = text.replace(/\n/g, '<br>');
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/^- (.+)$/gm, '• $1');
    text = text.replace(/^\d+\. (.+)$/gm, function(match, p1) {
        return '<br>' + match;
    });
    return text;
}

function handleChatKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatMessage();
    }
}

// ============================================================================
// QUICK ACTIONS
// ============================================================================

function askQuickQuestion(question) {
    const input = document.getElementById('chatInput');
    if (input) {
        input.value = question;
        sendChatMessage();
    }
}

function toggleChat() {
    const chatSection = document.getElementById('chat');
    const floatingBtn = document.getElementById('floatingChatBtn');
    
    if (chatSection && floatingBtn) {
        const isVisible = chatSection.style.display !== 'none';
        chatSection.style.display = isVisible ? 'none' : 'block';
        floatingBtn.style.display = isVisible ? 'block' : 'none';
    }
}

console.log('✅ Balaramaji Chat functions loaded');