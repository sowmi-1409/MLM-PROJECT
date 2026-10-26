/**
 * NewsGuard Main JavaScript
 * Handles theme toggling, mobile navigation, and session state.
 */

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initMobileNav();
  initSessionControls();
});

// Theme Management (Light / Dark)
function initTheme() {
  const root = document.documentElement;
  const themeToggle = document.getElementById('theme-toggle');
  const headerThemeToggle = document.getElementById('header-theme-toggle');
  const themeIcon = document.getElementById('theme-icon');
  const themeText = document.getElementById('theme-text');

  // Load saved preference or default to dark
  const savedTheme = localStorage.getItem('newsguard-theme') || 'dark';
  applyTheme(savedTheme);

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    localStorage.setItem('newsguard-theme', theme);
    if (themeIcon) {
      themeIcon.className = theme === 'dark' ? 'fa-solid fa-moon' : 'fa-solid fa-sun';
    }
    if (themeText) {
      themeText.textContent = theme === 'dark' ? 'Dark Mode' : 'Light Mode';
    }
  }

  function toggleTheme() {
    const currentTheme = root.getAttribute('data-theme') || 'dark';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme);
  }

  if (themeToggle) themeToggle.addEventListener('click', toggleTheme);
  if (headerThemeToggle) headerThemeToggle.addEventListener('click', toggleTheme);
}

// Mobile Sidebar Drawer Navigation
function initMobileNav() {
  const menuToggle = document.getElementById('menu-toggle');
  const sidebar = document.getElementById('sidebar');

  if (menuToggle && sidebar) {
    menuToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      sidebar.classList.toggle('open');
    });

    document.addEventListener('click', (e) => {
      if (sidebar.classList.contains('open') && !sidebar.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });
  }
}

// Session Statistics Controls
function initSessionControls() {
  const clearBtn = document.getElementById('btn-clear-session');
  if (clearBtn) {
    clearBtn.addEventListener('click', async () => {
      if (!confirm('Are you sure you want to reset all session analysis metrics?')) return;
      try {
        const resp = await fetch('/api/clear-session', { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
          window.location.reload();
        }
      } catch (err) {
        console.error('Failed to reset session:', err);
      }
    });
  }
}
