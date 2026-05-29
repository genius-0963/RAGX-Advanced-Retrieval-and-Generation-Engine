// Initialize Lucide icons
lucide.createIcons();

// Theme Logic
const themeToggle = document.getElementById('theme-toggle');
const currentTheme = localStorage.getItem('theme') || 'light';
document.documentElement.setAttribute('data-theme', currentTheme);
updateThemeIcon(currentTheme);

themeToggle.addEventListener('click', () => {
    let theme = document.documentElement.getAttribute('data-theme');
    theme = theme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    updateThemeIcon(theme);
});

function updateThemeIcon(theme) {
    if (theme === 'dark') {
        themeToggle.innerHTML = '<i data-lucide="sun"></i>';
    } else {
        themeToggle.innerHTML = '<i data-lucide="moon"></i>';
    }
    lucide.createIcons({ root: themeToggle });
}

lucide.createIcons();

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const fileUpload = document.getElementById('file-upload');
const dropZone = document.getElementById('drop-zone');
const uploadStatus = document.getElementById('upload-status');
const documentList = document.getElementById('document-list');

// Session State
let sessionId = 'session_' + Math.random().toString(36).substring(2, 11);

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    fetchDocuments();
    
    // Auto-resize textarea
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value.trim() === '') {
            sendBtn.disabled = true;
        } else {
            sendBtn.disabled = false;
        }
    });

    // Enter to send
    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (this.value.trim() !== '') {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });
});

// --- Chat Logic ---
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    // Reset input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Add User Message
    appendMessage('user', message);

    // Add Loading Message
    const loadingId = appendLoading();

    try {
        const response = await fetch('/api/v1/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId
            })
        });

        const data = await response.json();
        removeMessage(loadingId);

        if (response.ok) {
            appendMessage('bot', data.answer, data.sources);
        } else {
            appendMessage('bot', `Error: ${data.detail || 'Failed to get a response.'}`);
        }
    } catch (error) {
        removeMessage(loadingId);
        appendMessage('bot', 'Network error. Please make sure the API is running.');
    }
});

function appendMessage(role, text, sources = []) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role === 'user' ? 'user-msg' : 'system-msg'}`;
    
    const icon = role === 'user' ? 'user' : 'cpu';
    
    let sourcesHTML = '';
    if (sources && sources.length > 0) {
        sourcesHTML = '<div class="citations">';
        // Deduplicate sources by name for UI simplicity
        const uniqueSources = [...new Set(sources.map(s => {
            // Extract filename from path
            const parts = s.source.split(/[/\\]/);
            return parts[parts.length - 1];
        }))];
        
        uniqueSources.forEach(src => {
            sourcesHTML += `<span class="citation-tag"><i data-lucide="file-text"></i> ${src}</span>`;
        });
        sourcesHTML += '</div>';
    }

    msgDiv.innerHTML = `
        <div class="avatar"><i data-lucide="${icon}"></i></div>
        <div class="message-content">
            <p>${text.replace(/\\n/g, '<br>')}</p>
            ${sourcesHTML}
        </div>
    `;

    chatMessages.appendChild(msgDiv);
    lucide.createIcons({ root: msgDiv });
    scrollToBottom();
}

function appendLoading() {
    const id = 'loading-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.id = id;
    msgDiv.className = `message system-msg`;
    msgDiv.innerHTML = `
        <div class="avatar"><i data-lucide="cpu"></i></div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(msgDiv);
    lucide.createIcons({ root: msgDiv });
    scrollToBottom();
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// --- File Upload Logic ---
dropZone.addEventListener('click', () => fileUpload.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFileUpload(e.dataTransfer.files);
    }
});

fileUpload.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileUpload(e.target.files);
    }
});

async function handleFileUpload(files) {
    const formData = new FormData();
    
    // API supports batch if we used the batch route, but we'll loop or use /batch
    // Let's use the batch endpoint for multiple files
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    uploadStatus.innerHTML = `<span style="color: var(--text-secondary)">Uploading ${files.length} file(s)...</span>`;

    try {
        const response = await fetch('/api/v1/ingest/batch', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (response.ok) {
            uploadStatus.innerHTML = `<span style="color: #10b981"><i data-lucide="check-circle" style="width: 14px; height:14px; display:inline"></i> Queued for ingestion</span>`;
            // Refresh document list after a short delay to allow background processing
            setTimeout(fetchDocuments, 3000);
        } else {
            uploadStatus.innerHTML = `<span style="color: #ef4444">Upload failed: ${data.detail}</span>`;
        }
    } catch (error) {
        uploadStatus.innerHTML = `<span style="color: #ef4444">Network error during upload.</span>`;
    }
    
    lucide.createIcons({ root: uploadStatus });
    fileUpload.value = ''; // reset
}

async function fetchDocuments() {
    try {
        const response = await fetch('/api/v1/documents');
        const data = await response.json();
        
        documentList.innerHTML = '';
        if (data.documents && data.documents.length > 0) {
            data.documents.forEach(doc => {
                // Get filename from path
                const parts = doc.source.split(/[/\\]/);
                const filename = parts[parts.length - 1];
                
                const li = document.createElement('li');
                li.className = 'document-item';
                li.innerHTML = `
                    <i data-lucide="file"></i>
                    <span style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 180px;" title="${filename}">
                        ${filename}
                    </span>
                    <span style="margin-left: auto; font-size: 0.7rem; color: var(--primary)">${doc.chunk_count} chunks</span>
                `;
                documentList.appendChild(li);
            });
            lucide.createIcons({ root: documentList });
        } else {
            documentList.innerHTML = '<li style="color: var(--text-secondary); font-size: 0.8rem; text-align: center; padding: 20px 0;">No documents found.</li>';
        }
    } catch (error) {
        console.error("Failed to fetch documents", error);
    }
}
