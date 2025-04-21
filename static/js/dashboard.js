// Dashboard navigation handling
document.addEventListener('DOMContentLoaded', function() {
    // Toggle sidebar on mobile
    const sidebarToggler = document.getElementById('sidebar-toggler');
    if (sidebarToggler) {
        sidebarToggler.addEventListener('click', function() {
            document.querySelector('.dashboard-container').classList.toggle('sidebar-collapsed');
        });
    }
    
    // Coba untuk memuat data tersimpan saat halaman pertama kali dibuka
    loadSavedData();
    
    // Event listener untuk tombol scrape
    const scrapeButton = document.getElementById('scrape-button');
    if (scrapeButton) {
        scrapeButton.addEventListener('click', handleScrapeRequest);
    }
    
    // Navigation handling - show active content based on navigation
    const navLinks = document.querySelectorAll('.sidebar-nav .nav-link');
    const contentSections = document.querySelectorAll('.content-section');
    
    function showSection(sectionId) {
        // Hide all sections
        contentSections.forEach(section => {
            section.style.display = 'none';
        });
        
        // Show selected section
        const activeSection = document.getElementById(sectionId);
        if (activeSection) {
            activeSection.style.display = 'block';
        }
        
        // Update active navigation link
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('data-section') === sectionId) {
                link.classList.add('active');
            }
        });
        
        // Special handling for scrape section - auto show results if we have data
        if (sectionId === 'scrape-section' && window.reviewsData && window.reviewsData.length > 0) {
            const resultsContainer = document.getElementById('results-container');
            if (resultsContainer) {
                resultsContainer.style.display = 'block';
            }
        }
    }
    
    // Set up click handlers for navigation
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const sectionId = this.getAttribute('data-section');
            showSection(sectionId);
            
            // Update URL hash
            window.location.hash = sectionId;
        });
    });
    
    // Check for hash in URL
    if (window.location.hash) {
        const sectionId = window.location.hash.substring(1);
        const matchingLink = document.querySelector(`.nav-link[data-section="${sectionId}"]`);
        if (matchingLink) {
            matchingLink.click();
        } else {
            showSection('home-section');
        }
    } else {
        showSection('home-section');
    }
});

// Function to get confidence bar color based on confidence value
function getConfidenceBarColor(confidence) {
    if (confidence < 0.3) {
        return '#f44336'; // Red
    } else if (confidence < 0.7) {
        return '#ff9800'; // Orange
    } else {
        return '#4caf50'; // Green
    }
}

// Update filter status counter
function updateFilterStatus(filteredCount, totalCount) {
    const filterStatusElem = document.getElementById('filter-status');
    if (filterStatusElem) {
        filterStatusElem.textContent = `Menampilkan ${filteredCount} dari ${totalCount} ulasan`;
    }
}

// Initialize date range picker
function initDateRangePicker(reviewsData) {
    const dateStartPicker = document.getElementById('date-start');
    const dateEndPicker = document.getElementById('date-end');
    
    if (!dateStartPicker || !dateEndPicker || !reviewsData || !reviewsData.length) {
        return;
    }
    
    // Get earliest and latest dates from reviews
    let earliestDate = new Date();
    let latestDate = new Date(0);
    
    reviewsData.forEach(review => {
        // Extract date from all possible sources
        let reviewDate;
        
        // Try different date sources in order of preference - using the same logic as in filtering
        if (typeof review.date === 'string' && review.date) {
            reviewDate = new Date(review.date);
        } else if (review.date && review.date.year && review.date.month && review.date.day) {
            reviewDate = new Date(review.date.year, review.date.month - 1, review.date.day);
        } else if (review.at) {
            try {
                // The 'at' field can be ISO date string or Unix timestamp
                if (typeof review.at === 'string' && review.at.includes('T')) {
                    // This is an ISO date string like "2018-03-15T12:19:45"
                    reviewDate = new Date(review.at);
                } else if (!isNaN(review.at)) {
                    // This is a Unix timestamp (seconds)
                    reviewDate = new Date(review.at * 1000);
                } else {
                    // Try direct parsing in case it's another date format
                    reviewDate = new Date(review.at);
                }
            } catch (e) {
                console.error('Error parsing date from review.at:', e, review.at);
            }
        }
        
        if (reviewDate && !isNaN(reviewDate.getTime())) {
            if (reviewDate < earliestDate) {
                earliestDate = reviewDate;
            }
            if (reviewDate > latestDate) {
                latestDate = reviewDate;
            }
        }
    });
    
    console.log('Date range found:', { earliest: earliestDate, latest: latestDate });
    
    // Format date for display
    function formatDate(date) {
        return date.toISOString().split('T')[0];
    }
    
    // Set default values according to user requirements:
    // - Start date should be the earliest date from the scraped reviews
    // - End date should be today's date
    
    // Make sure we have a valid earliest date
    // (If there are no valid dates, use a reasonable default)
    const today = new Date();
    
    if (earliestDate > today || earliestDate > latestDate) {
        // If earliest date is in the future or greater than latest review date
        // it means we didn't find any valid dates in the reviews
        // Use the oldest scraped review date or a fallback date from 3 months ago
        if (latestDate < today && latestDate.getTime() > new Date(0).getTime()) {
            earliestDate = latestDate;
        } else {
            // Fallback: 3 months ago
            earliestDate = new Date(today);
            earliestDate.setMonth(today.getMonth() - 3);
        }
    }
    
    // Set the date values
    dateStartPicker.value = formatDate(earliestDate);
    dateEndPicker.value = formatDate(today);
    
    // Add event listeners
    dateStartPicker.addEventListener('change', displayFilteredReviews);
    dateEndPicker.addEventListener('change', displayFilteredReviews);
    
    // Set placeholders
    dateStartPicker.setAttribute('placeholder', formatDate(earliestDate));
    dateEndPicker.setAttribute('placeholder', formatDate(latestDate));
}

// Handle Scraping Google Play Store reviews
function handleScrapeRequest() {
    console.log('Memulai handleScrapeRequest');
    
    // Nonaktifkan tombol untuk mencegah klik ganda
    const scrapeButton = document.getElementById('scrape-button');
    if (scrapeButton) scrapeButton.disabled = true;
    
    // Ambil nilai max_reviews, gunakan 0 jika kosong (0 berarti ambil semua ulasan yang tersedia)
    const maxReviewsInput = document.getElementById('max-reviews');
    const maxReviews = maxReviewsInput ? (maxReviewsInput.value || '0') : '0';
    console.log('Max reviews:', maxReviews);
    
    // Tampilkan loading state
    document.querySelectorAll('.loading').forEach(el => el.style.display = 'flex');
    const statusText = document.getElementById('status-text');
    const progressContainer = document.getElementById('progress-container');
    const resultsContainer = document.getElementById('results-container');
    
    if (statusText) statusText.textContent = 'Memeriksa data tersimpan...';
    if (progressContainer) progressContainer.style.display = 'block';
    if (resultsContainer) resultsContainer.style.display = 'none';
    
    // Selalu melakukan scraping baru (tidak menggunakan data tersimpan) 
    // jika max_reviews adalah 0 atau jika pengguna memaksa scraping baru
    const forceNewScrape = maxReviews === '0';
    
    // Pertama, cek apakah ada data tersimpan
    fetch('/api/get_saved_data')
        .then(response => response.json())
        .then(savedData => {
            // Jika ada data tersimpan dan tidak diminta untuk memaksa scraping baru
            if (savedData && savedData.results && savedData.results.length > 0 && !forceNewScrape) {
                console.log('Menggunakan data tersimpan:', savedData.results.length, 'ulasan');
                if (statusText) statusText.textContent = 'Menggunakan data tersimpan...';
                
                // Tunggu sebentar untuk UI feedback
                setTimeout(() => {
                    // Hide loading dan tampilkan hasil
                    document.querySelectorAll('.loading').forEach(el => el.style.display = 'none');
                    if (progressContainer) progressContainer.style.display = 'none';
                    if (resultsContainer) resultsContainer.style.display = 'block';
                    
                    // Store data for filtering
                    window.reviewsData = savedData.results;
                    console.log('Data tersimpan dimuat ke window.reviewsData');
                    
                    // Initialize date range picker dengan data
                    initDateRangePicker(savedData.results);
                    
                    // Initialize aspect category filters
                    initAspectCategoryFilters(savedData.results);
                    
                    // Display all reviews
                    displayFilteredReviews();
                    console.log('Menampilkan ulasan terfilter dari data tersimpan');
                    
                    // Update summary stats
                    updateSummaryStats(savedData.results);
                    console.log('Statistik diperbarui dari data tersimpan');
                    
                    // Re-enable scrape button
                    if (scrapeButton) scrapeButton.disabled = false;
                    console.log('Proses selesai, tombol diaktifkan kembali');
                }, 500);
            } else {
                // Tidak ada data tersimpan atau diminta untuk memaksa scraping baru
                if (statusText) statusText.textContent = 'Melakukan scraping ulasan baru...';
                console.log('Memulai scraping baru, maxReviews:', maxReviews, 'forceNewScrape:', forceNewScrape);
                
                // Lakukan scraping dan analisis
                fetch('/api/scrape', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        max_reviews: parseInt(maxReviews),
                        force_new: forceNewScrape
                    })
                })
                .then(response => {
                    console.log('Response status:', response.status);
                    if (!response.ok) {
                        throw new Error('Network response was not ok: ' + response.statusText);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Data diterima dari scraping:', data);
                    
                    // Validasi data
                    if (!data || !data.results || !Array.isArray(data.results)) {
                        console.error('Format data tidak valid:', data);
                        throw new Error('Format data tidak valid');
                    }
                    
                    console.log(`Menerima ${data.results.length} ulasan dari API`);
                    
                    // Hide loading dan tampilkan hasil
                    document.querySelectorAll('.loading').forEach(el => el.style.display = 'none');
                    if (progressContainer) progressContainer.style.display = 'none';
                    if (resultsContainer) resultsContainer.style.display = 'block';
                    
                    // Store data for filtering
                    window.reviewsData = data.results;
                    console.log('Data scraping disimpan ke window.reviewsData');
                    
                    // Initialize date range picker dengan data baru
                    initDateRangePicker(data.results);
                    
                    // Initialize aspect category filters dengan data baru
                    initAspectCategoryFilters(data.results);
                    
                    // Display all reviews
                    displayFilteredReviews();
                    console.log('Menampilkan ulasan terfilter');
                    
                    // Update summary stats
                    updateSummaryStats(data.results);
                    console.log('Statistik diperbarui');
                    
                    // Re-enable scrape button
                    if (scrapeButton) scrapeButton.disabled = false;
                    console.log('Proses selesai, tombol diaktifkan kembali');
                })
                .catch(error => {
                    console.error('Error selama scraping:', error);
                    document.querySelectorAll('.loading').forEach(el => el.style.display = 'none');
                    if (statusText) statusText.textContent = 'Terjadi kesalahan: ' + error.message;
                    if (progressContainer) progressContainer.style.display = 'none';
                    if (scrapeButton) scrapeButton.disabled = false;
                    alert('Terjadi kesalahan saat melakukan scraping: ' + error.message);
                });
            }
        })
        .catch(error => {
            console.error('Error ketika mengecek data tersimpan:', error);
            document.querySelectorAll('.loading').forEach(el => el.style.display = 'none');
            if (statusText) statusText.textContent = 'Terjadi kesalahan: ' + error.message;
            if (progressContainer) progressContainer.style.display = 'none';
            if (scrapeButton) scrapeButton.disabled = false;
            alert('Terjadi kesalahan saat memeriksa data tersimpan: ' + error.message);
        });
}

// Function to get aspect categories from reviews
function initAspectCategoryFilters(reviewsData) {
    const categoryContainer = document.getElementById('aspect-category-filters');
    if (!categoryContainer || !reviewsData) return;
    
    const categories = new Set();
    
    // Extract all unique categories from all triplets
    reviewsData.forEach(review => {
        if (Array.isArray(review.triplets)) {
            review.triplets.forEach(triplet => {
                if (triplet.aspect_category && triplet.aspect_category !== 'Unknown') {
                    categories.add(triplet.aspect_category);
                }
            });
        }
    });
    
    // Sort categories
    const sortedCategories = Array.from(categories).sort();
    
    // Clear loading spinner
    categoryContainer.innerHTML = '';
    
    // If no categories
    if (sortedCategories.length === 0) {
        categoryContainer.innerHTML = '<span class="text-muted">Tidak ada kategori tersedia</span>';
        return;
    }
    
    // Add select/deselect all control
    const selectAllContainer = document.createElement('div');
    selectAllContainer.className = 'form-check mb-2';
    selectAllContainer.innerHTML = `
        <input class="form-check-input" type="checkbox" id="select-all-categories" checked>
        <label class="form-check-label" for="select-all-categories">Pilih Semua</label>
    `;
    categoryContainer.appendChild(selectAllContainer);
    
    // Add divider
    const divider = document.createElement('hr');
    divider.className = 'my-2';
    categoryContainer.appendChild(divider);
    
    // Add categories container
    const categoriesDiv = document.createElement('div');
    categoriesDiv.className = 'category-items';
    categoryContainer.appendChild(categoriesDiv);
    
    // Add option for each category
    sortedCategories.forEach(category => {
        const categoryBtn = document.createElement('div');
        categoryBtn.className = 'form-check form-check-inline category-option';
        categoryBtn.innerHTML = `
            <input class="form-check-input category-filter" type="checkbox" id="cat-${category.replace(/\s+/g, '-')}" value="${category}" checked>
            <label class="form-check-label" for="cat-${category.replace(/\s+/g, '-')}">${category}</label>
            <span class="only-button">Hanya</span>
        `;
        categoriesDiv.appendChild(categoryBtn);
        
        // Add styles for the only button
        const onlyButton = categoryBtn.querySelector('.only-button');
        onlyButton.style.display = 'none';
        onlyButton.style.marginLeft = '5px';
        onlyButton.style.fontSize = '0.75rem';
        onlyButton.style.color = '#0d6efd';
        onlyButton.style.cursor = 'pointer';
        onlyButton.style.fontWeight = 'bold';
        
        // Show/hide "Only" button on hover
        categoryBtn.addEventListener('mouseenter', function() {
            onlyButton.style.display = 'inline';
        });
        
        categoryBtn.addEventListener('mouseleave', function() {
            onlyButton.style.display = 'none';
        });
        
        // Add event listener to checkbox for normal filtering
        const checkbox = categoryBtn.querySelector('.category-filter');
        checkbox.addEventListener('change', function() {
            // Update "select all" checkbox
            updateSelectAllCheckbox();
            displayFilteredReviews();
        });
        
        // Add event listener to "Only" button
        onlyButton.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            // Uncheck all categories
            document.querySelectorAll('.category-filter').forEach(cb => {
                cb.checked = false;
            });
            
            // Check only this category
            checkbox.checked = true;
            
            // Update "select all" checkbox
            updateSelectAllCheckbox();
            
            // Update displayed reviews
            displayFilteredReviews();
        });
    });
    
    // Function to update "select all" checkbox state
    function updateSelectAllCheckbox() {
        const allCheckboxes = document.querySelectorAll('.category-filter');
        const checkedCheckboxes = document.querySelectorAll('.category-filter:checked');
        const selectAllCheckbox = document.getElementById('select-all-categories');
        
        if (selectAllCheckbox) {
            selectAllCheckbox.checked = allCheckboxes.length === checkedCheckboxes.length;
            selectAllCheckbox.indeterminate = checkedCheckboxes.length > 0 && checkedCheckboxes.length < allCheckboxes.length;
        }
    }
    
    // Add event listener to "select all" checkbox
    const selectAllCheckbox = document.getElementById('select-all-categories');
    if (selectAllCheckbox) {
        selectAllCheckbox.addEventListener('change', function() {
            const isChecked = this.checked;
            document.querySelectorAll('.category-filter').forEach(checkbox => {
                checkbox.checked = isChecked;
            });
            displayFilteredReviews();
        });
    }
}

// Display filtered reviews
function displayFilteredReviews() {
    if (!window.reviewsData) return;
    
    // Get filter values
    const dateStart = document.getElementById('date-start')?.value ? new Date(document.getElementById('date-start').value) : null;
    const dateEnd = document.getElementById('date-end')?.value ? new Date(document.getElementById('date-end').value) : null;
    if (dateEnd) dateEnd.setHours(23, 59, 59, 999); // End of day
    
    // Get selected categories
    const selectedCategories = [];
    document.querySelectorAll('#aspect-category-filters .category-filter:checked').forEach(checkbox => {
        selectedCategories.push(checkbox.value);
    });
    
    // Get sentiment filters
    const showPositive = document.getElementById('show-positive')?.checked !== false;
    const showNegative = document.getElementById('show-negative')?.checked !== false;
    const showNeutral = document.getElementById('show-neutral')?.checked !== false;
    
    // Flag for applying filtered triplets to reviews
    const applyFilteredTripletsToReviews = true;
    
    // Log filter values for debugging
    console.log('Filter dates:', { dateStart: dateStart ? dateStart.toISOString() : null, dateEnd: dateEnd ? dateEnd.toISOString() : null });
    
    // First, filter reviews by date
    let filteredReviews = window.reviewsData.filter(review => {
        // Filter by date
        if (dateStart || dateEnd) {
            let reviewDate;
            
            // Check all possible date formats
            if (typeof review.date === 'string' && review.date) {
                reviewDate = new Date(review.date);
            } else if (review.date && review.date.year && review.date.month && review.date.day) {
                reviewDate = new Date(review.date.year, review.date.month - 1, review.date.day);
            } else if (review.at) {
                try {
                    // The 'at' field can be ISO date string or Unix timestamp
                    // Check if it's already an ISO format string
                    if (typeof review.at === 'string' && review.at.includes('T')) {
                        // This is an ISO date string like "2018-03-15T12:19:45"
                        reviewDate = new Date(review.at);
                    } else if (!isNaN(review.at)) {
                        // This is a Unix timestamp (seconds)
                        reviewDate = new Date(review.at * 1000);
                    } else {
                        // Try direct parsing in case it's another date format
                        reviewDate = new Date(review.at);
                    }
                } catch (e) {
                    console.error('Error parsing date from review.at:', e, review.at);
                }
            }
            
            // Debug output
            console.log(`Review date for ID ${review.review_id}:`, {
                date: review.date,
                parsedDate: reviewDate,
                at: review.at,
                valid: reviewDate && !isNaN(reviewDate.getTime())
            });
            
            if (!reviewDate || isNaN(reviewDate.getTime())) return false;
            
            if (dateStart && reviewDate < dateStart) return false;
            if (dateEnd && reviewDate > dateEnd) return false;
        }
        
        return true; // Pass date filter
    });
    
    // Then, apply filter to triplets
    if (applyFilteredTripletsToReviews) {
        // Create a new array of reviews with filtered triplets
        filteredReviews = filteredReviews.map(review => {
            // Clone the review to avoid modifying the original data
            const filteredReview = { ...review };
            
            if (Array.isArray(review.triplets) && review.triplets.length > 0) {
                // Only include triplets that match the category and sentiment filters
                filteredReview.triplets = review.triplets.filter(triplet => {
                    // Category filter
                    if (selectedCategories.length > 0 && triplet.aspect_category) {
                        if (!selectedCategories.includes(triplet.aspect_category)) {
                            return false; // Skip triplets with non-selected categories
                        }
                    }
                    
                    // Sentiment filter
                    if (triplet.sentiment === 'POS' && !showPositive) return false;
                    if (triplet.sentiment === 'NEG' && !showNegative) return false;
                    if (triplet.sentiment === 'NEU' && !showNeutral) return false;
                    
                    return true; // Pass all filters
                });
            }
            
            return filteredReview;
        }).filter(review => {
            // Only keep reviews that have at least one matching triplet after filtering
            return Array.isArray(review.triplets) && review.triplets.length > 0;
        });
    } else {
        // Old behavior - Filter by has triplets (at least a triplet with matching category and sentiment)
        filteredReviews = filteredReviews.filter(review => {
            if (Array.isArray(review.triplets) && review.triplets.length > 0) {
                return review.triplets.some(triplet => {
                    // Category filter
                    if (selectedCategories.length > 0 && triplet.aspect_category) {
                        if (!selectedCategories.includes(triplet.aspect_category)) {
                            return false;
                        }
                    }
                    
                    // Sentiment filter
                    if (triplet.sentiment === 'POS' && !showPositive) return false;
                    if (triplet.sentiment === 'NEG' && !showNegative) return false;
                    if (triplet.sentiment === 'NEU' && !showNeutral) return false;
                    
                    return true;
                });
            }
            return false;
        });
    }
    
    // Update filter status
    updateFilterStatus(filteredReviews.length, window.reviewsData.length);
    
    // Display reviews
    const reviewsContainer = document.getElementById('reviews-container');
    if (!reviewsContainer) return;
    
    reviewsContainer.innerHTML = '';
    
    if (filteredReviews.length === 0) {
        reviewsContainer.innerHTML = '<p class="text-center text-muted">Tidak ada ulasan yang sesuai filter.</p>';
        return;
    }
    
    // Sort reviews by date (newest first)
    filteredReviews.sort((a, b) => {
        let dateA, dateB;
        
        if (typeof a.date === 'string') {
            dateA = new Date(a.date);
        } else if (a.date && a.date.year && a.date.month && a.date.day) {
            dateA = new Date(a.date.year, a.date.month - 1, a.date.day);
        }
        
        if (typeof b.date === 'string') {
            dateB = new Date(b.date);
        } else if (b.date && b.date.year && b.date.month && b.date.day) {
            dateB = new Date(b.date.year, b.date.month - 1, b.date.day);
        }
        
        if (!dateA || isNaN(dateA.getTime())) return 1;
        if (!dateB || isNaN(dateB.getTime())) return -1;
        
        return dateB - dateA;
    });
    
    // Display reviews
    filteredReviews.forEach(review => {
        // Format date - handle various date formats using the same logic as the filter
        let formattedDate = 'Tanggal tidak diketahui';
        let reviewDate;
        
        // Try different date sources in order of preference
        if (typeof review.date === 'string' && review.date) {
            reviewDate = new Date(review.date);
        } else if (review.date && review.date.year && review.date.month && review.date.day) {
            reviewDate = new Date(review.date.year, review.date.month - 1, review.date.day);
        } else if (review.at) {
            // Use the same logic as in the filter function
            try {
                // The 'at' field can be ISO date string or Unix timestamp
                if (typeof review.at === 'string' && review.at.includes('T')) {
                    // This is an ISO date string like "2018-03-15T12:19:45"
                    reviewDate = new Date(review.at);
                } else if (!isNaN(review.at)) {
                    // This is a Unix timestamp (seconds)
                    reviewDate = new Date(review.at * 1000);
                } else {
                    // Try direct parsing in case it's another date format
                    reviewDate = new Date(review.at);
                }
            } catch (e) {
                console.error('Error formatting date from review.at:', e, review.at);
            }
        }
        
        // Format the date if valid
        if (reviewDate && !isNaN(reviewDate.getTime())) {
            try {
                formattedDate = reviewDate.toLocaleDateString('id-ID', {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                });
            } catch (e) {
                console.error('Error formatting date for display:', e);
                // Fallback to simple format
                formattedDate = reviewDate.toLocaleDateString();
            }
        }
        
        // Create stars HTML
        let starsHtml = '';
        if (review.score) {
            const score = parseInt(review.score);
            for (let i = 1; i <= 5; i++) {
                if (i <= score) {
                    starsHtml += '<i class="bi bi-star-fill text-warning"></i>';
                } else {
                    starsHtml += '<i class="bi bi-star text-muted"></i>';
                }
            }
        }
        
        const reviewCard = document.createElement('div');
        reviewCard.className = 'card review-card mb-3';
        
        // Generate triplets HTML
        let tripletsHtml = '';
        if (!Array.isArray(review.triplets) || review.triplets.length === 0) {
            tripletsHtml = `
                <div class="no-results mt-3">
                    <p class="text-muted">Tidak ditemukan triplet sentimen pada ulasan ini.</p>
                    <p class="text-muted small">Kemungkinan ulasan terlalu pendek atau tidak mengandung sentimen yang jelas.</p>
                </div>
            `;
        } else {
            tripletsHtml = '<div class="mt-3">';
            // Display results title with triplet count
            tripletsHtml += `<h6 class="mb-2">Hasil Prediksi: <span class="text-muted">(${review.triplets.length} triplet)</span></h6>`;
            
            review.triplets.forEach(triplet => {
                let sentimentClass = '';
                let sentimentLabel = '';
                if (triplet.sentiment === 'POS') {
                    sentimentClass = 'positive';
                    sentimentLabel = 'Positif';
                } else if (triplet.sentiment === 'NEG') {
                    sentimentClass = 'negative';
                    sentimentLabel = 'Negatif';
                } else {
                    sentimentClass = 'neutral';
                    sentimentLabel = 'Netral';
                }
                
                tripletsHtml += `
                    <div class="card triplet-card ${sentimentClass} mt-2">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start mb-3">
                                <div class="d-flex align-items-center">
                                    <span class="sentiment-badge ${sentimentClass} me-2">${sentimentLabel}</span>
                                    ${triplet.aspect_category ? `<span class="aspect-category-badge">${triplet.aspect_category}</span>` : ''}
                                </div>
                                ${triplet.aspect_category ? `
                                <div class="confidence-container text-end">
                                    <div class="confidence-label small">Category Confidence: ${triplet.category_confidence ? (triplet.category_confidence * 100).toFixed(1) : '0.0'}%</div>
                                    <div class="confidence-bar-bg">
                                        <div class="confidence-bar-inner" style="width: ${triplet.category_confidence ? (triplet.category_confidence * 100) : 0}%; background-color: ${getConfidenceBarColor(triplet.category_confidence || 0)}"></div>
                                    </div>
                                </div>
                                ` : ''}
                            </div>
                            <div class="triplet-content">
                                <div class="mb-2">
                                    <span class="aspect-label">Aspek:</span>
                                    <span class="aspect-value">${triplet.aspect_term}</span>
                                </div>
                                <div>
                                    <span class="opinion-label">Opini:</span>
                                    <span class="opinion-value">${triplet.opinion_term}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            tripletsHtml += '</div>';
        }
        
        reviewCard.innerHTML = `
            <div class="card-header">
                <div class="d-flex justify-content-between align-items-center">
                    <div class="review-score">
                        ${starsHtml} <span class="ms-1">${review.score}/5</span>
                    </div>
                    <div class="review-date">
                        <i class="bi bi-calendar3 me-1"></i>${formattedDate}
                    </div>
                </div>
            </div>
            <div class="card-body">
                <div class="review-text mb-3">${review.review_text}</div>
                ${tripletsHtml}
            </div>
        `;
        
        reviewsContainer.appendChild(reviewCard);
    });
}

// Update summary stats
function updateSummaryStats(reviewsData) {
    if (!reviewsData || reviewsData.length === 0) return;
    
    // Count sentiments
    let positiveCount = 0;
    let negativeCount = 0;
    let neutralCount = 0;
    let totalTriplets = 0;
    
    // Count aspect categories
    const categoryCount = {};
    
    reviewsData.forEach(review => {
        if (Array.isArray(review.triplets)) {
            review.triplets.forEach(triplet => {
                totalTriplets++;
                
                // Count sentiment
                if (triplet.sentiment === 'POS') positiveCount++;
                else if (triplet.sentiment === 'NEG') negativeCount++;
                else neutralCount++;
                
                // Count category
                if (triplet.aspect_category) {
                    if (!categoryCount[triplet.aspect_category]) {
                        categoryCount[triplet.aspect_category] = 0;
                    }
                    categoryCount[triplet.aspect_category]++;
                }
            });
        }
    });
    
    // Update sentiment chart in home section
    updateSentimentChart('sentiment-chart', positiveCount, negativeCount, neutralCount);
    
    // Update sentiment chart in scrape section (jika ada)
    updateSentimentChart('sentiment-chart-scrape', positiveCount, negativeCount, neutralCount);
    
    // Update category chart in home section
    updateCategoryChart('category-chart', categoryCount);
    
    // Update category chart in scrape section (jika ada)
    updateCategoryChart('category-chart-scrape', categoryCount);
    
    // Update summary numbers
    const totalReviewsElement = document.getElementById('total-reviews');
    const totalTripletsElement = document.getElementById('total-triplets');
    const positiveSentimentElement = document.getElementById('positive-sentiment');
    const neutralSentimentElement = document.getElementById('neutral-sentiment');
    const negativeSentimentElement = document.getElementById('negative-sentiment');
    
    if (totalReviewsElement) totalReviewsElement.textContent = reviewsData.length;
    if (totalTripletsElement) totalTripletsElement.textContent = totalTriplets;
    if (positiveSentimentElement) positiveSentimentElement.textContent = positiveCount;
    if (neutralSentimentElement) neutralSentimentElement.textContent = neutralCount;
    if (negativeSentimentElement) negativeSentimentElement.textContent = negativeCount;
}

// Function to update home page statistics
function updateHomePageStats(reviewsData) {
    // This function is just a wrapper around updateSummaryStats which already updates the home page stats
    updateSummaryStats(reviewsData);
    
    // Make sure home section has visible stats
    const homeChartElements = document.querySelectorAll('#home-section .chart-container');
    homeChartElements.forEach(el => {
        if (el) el.style.display = 'block';
    });
    
    // Show home stats containers
    const homeStatsContainers = document.querySelectorAll('#home-section .stats-container');
    homeStatsContainers.forEach(container => {
        if (container) container.style.display = 'block';
    });
}

// Fungsi helper untuk update sentiment chart
function updateSentimentChart(chartId, positiveCount, negativeCount, neutralCount) {
    const sentimentCtx = document.getElementById(chartId);
    if (!sentimentCtx) return;
    
    // Clear existing chart if any
    if (window[chartId + 'Instance']) {
        window[chartId + 'Instance'].destroy();
    }
    
    window[chartId + 'Instance'] = new Chart(sentimentCtx, {
        type: 'doughnut',
        data: {
            labels: ['Positif', 'Negatif', 'Netral'],
            datasets: [{
                data: [positiveCount, negativeCount, neutralCount],
                backgroundColor: [
                    '#4caf50', // Green
                    '#f44336', // Red
                    '#ff9800'  // Orange
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#333',
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = positiveCount + negativeCount + neutralCount;
                            const percentage = Math.round((context.raw / total) * 100);
                            return `${context.label}: ${context.raw} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
} // Close updateSentimentChart function

// Fungsi helper untuk update category chart
function updateCategoryChart(chartId, categoryCount) {
    const categoryCtx = document.getElementById(chartId);
    if (!categoryCtx) return;
    
    // Clear existing chart if any
    if (window[chartId + 'Instance']) {
        window[chartId + 'Instance'].destroy();
    }
    
    // Sort categories by count (descending)
    const sortedCategories = Object.entries(categoryCount)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5); // Only show top 5
    
    const labels = sortedCategories.map(cat => cat[0]);
    const values = sortedCategories.map(cat => cat[1]);
    
    // Create color gradient based on number of categories
    const colors = labels.map((_, index) => {
        const hue = 215 + index * (120 / labels.length);
        return `hsl(${hue}, 70%, 60%)`;
    });
    
    window[chartId + 'Instance'] = new Chart(categoryCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Jumlah',
                data: values,
                backgroundColor: colors,
                borderColor: colors.map(color => color.replace('60%', '50%')),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = values.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((context.raw / total) * 100);
                            return `${context.label}: ${context.raw} (${percentage}%)`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// Text prediction functionality
function handleTextPrediction(event) {
    event.preventDefault();
    
    const reviewText = document.getElementById('review-text').value.trim();
    if (!reviewText) return;
    
    // Show loading spinner
    document.querySelector('#predict-text-section .loading').style.display = 'flex';
    document.getElementById('prediction-results').style.display = 'none';
    
    // API request to predict
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: reviewText
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Hide loading spinner
        document.querySelector('#predict-text-section .loading').style.display = 'none';
        document.getElementById('prediction-results').style.display = 'block';
        
        const tripletsContainer = document.getElementById('triplets-container');
        const noTripletsMessage = document.getElementById('no-triplets');
        
        // Clear previous results
        tripletsContainer.innerHTML = '';
        
        // Check if there are triplets
        if (!data.triplets || data.triplets.length === 0) {
            noTripletsMessage.style.display = 'block';
            return;
        }
        
        // Hide no triplets message
        noTripletsMessage.style.display = 'none';
        
        // Display triplets
        data.triplets.forEach(triplet => {
            // Create sentiment label
            let sentimentClass = '';
            let sentimentLabel = '';
            if (triplet.sentiment === 'POS') {
                sentimentClass = 'positive';
                sentimentLabel = 'Positif';
            } else if (triplet.sentiment === 'NEG') {
                sentimentClass = 'negative';
                sentimentLabel = 'Negatif';
            } else {
                sentimentClass = 'neutral';
                sentimentLabel = 'Netral';
            }
            
            // Get confidence color if needed for styling
            const confidenceColor = getConfidenceBarColor(triplet.confidence);
            
            const tripletCard = document.createElement('div');
            tripletCard.className = `card triplet-card ${sentimentClass} mt-2`;
            tripletCard.innerHTML = `
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start mb-3">
                        <div class="d-flex align-items-center">
                            <span class="sentiment-badge ${sentimentClass} me-2">${sentimentLabel}</span>
                            ${triplet.aspect_category ? `<span class="aspect-category-badge">${triplet.aspect_category}</span>` : ''}
                        </div>
                        ${triplet.aspect_category ? `
                        <div class="confidence-container text-end">
                            <div class="confidence-label small">Category Confidence: ${triplet.category_confidence ? (triplet.category_confidence * 100).toFixed(1) : '0.0'}%</div>
                            <div class="confidence-bar-bg" style="width: 100px;">
                                <div class="confidence-bar-inner" style="width: ${triplet.category_confidence ? (triplet.category_confidence * 100) : 0}%; background-color: ${getConfidenceBarColor(triplet.category_confidence || 0)}"></div>
                            </div>
                        </div>
                        ` : ''}
                    </div>
                    <div class="triplet-content">
                        <div class="mb-2">
                            <span class="aspect-label">Aspek:</span>
                            <span class="aspect-value">${triplet.aspect}</span>
                        </div>
                        <div>
                            <span class="opinion-label">Opini:</span>
                            <span class="opinion-value">${triplet.opinion}</span>
                        </div>

                    </div>
                </div>
            `;
            
            tripletsContainer.appendChild(tripletCard);
        });
    })
    .catch(error => {
        console.error('Error predicting text:', error);
        document.querySelector('#predict-text-section .loading').style.display = 'none';
        document.getElementById('prediction-results').style.display = 'block';
        document.getElementById('triplets-container').innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle-fill me-2"></i>
                Error: ${error.message}
            </div>
        `;
    });
}

// Fungsi untuk memuat data yang tersimpan
function loadSavedData() {
    // Hide loading states
    document.querySelectorAll('.loading').forEach(el => el.style.display = 'none');

    fetch('/api/get_saved_data')
        .then(response => response.json())
        .then(data => {
            if (data && data.results && data.results.length > 0) {
                // Store results for filtering
                window.reviewsData = data.results;
                
                // Initialize date range picker dengan data yang dimuat
                initDateRangePicker(data.results);
                
                // Initialize aspect category filters
                initAspectCategoryFilters(data.results);
                
                // Display all reviews
                displayFilteredReviews();
                
                // Update summary stats
                updateSummaryStats(data.results);
                
                // Show results container automatically
                const resultsContainer = document.getElementById('results-container');
                if (resultsContainer) {
                    resultsContainer.style.display = 'block';
                }

                // Update the home page stats dan pastikan semua elemen chart dan stats di beranda terlihat
                updateHomePageStats(data.results);
                
                // Force beranda menjadi section aktif jika tidak ada section yang aktif
                const activeSections = Array.from(document.querySelectorAll('.content-section')).filter(section => 
                    section.style.display === 'block' || section.style.display === '');
                
                if (activeSections.length === 0) {
                    showSection('home-section');
                    // Highlight the home nav link
                    document.querySelectorAll('.sidebar-nav .nav-link').forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('data-section') === 'home-section') {
                            link.classList.add('active');
                        }
                    });
                }
                
                console.log(`Loaded ${data.results.length} reviews from saved data`);
            } else {
                console.log('No saved data found');
            }
        })
        .catch(error => {
            console.error('Error loading saved data:', error);
        });
}
