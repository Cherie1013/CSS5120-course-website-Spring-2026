// ===== 主题切换 =====
(function() {
    const STORAGE_KEY = 'course-theme';
    const THEMES = [
        { key: 'light', label: 'Light', color: '#dbe4e7' },
        { key: 'dark',  label: 'Dark',  color: '#19546e' }
    ];

    function setTheme(themeKey) {
        document.documentElement.setAttribute('data-theme', themeKey);
        localStorage.setItem(STORAGE_KEY, themeKey);
        updatePickerState(themeKey);
    }

    function updatePickerState(activeKey) {
        document.querySelectorAll('.theme-dot').forEach(dot => {
            dot.classList.toggle('active', dot.dataset.theme === activeKey);
        });
    }

    function init() {
        const saved = localStorage.getItem(STORAGE_KEY) || 'light';
        setTheme(saved);

        // 展开/收起
        const picker = document.querySelector('.theme-picker');
        const trigger = document.querySelector('.theme-trigger');

        if (trigger && picker) {
            trigger.addEventListener('click', function(e) {
                e.stopPropagation();
                picker.classList.toggle('open');
            });

            // 点击外部收起
            document.addEventListener('click', function(e) {
                if (!picker.contains(e.target)) {
                    picker.classList.remove('open');
                }
            });
        }

        // 点击圆点切换主题
        document.querySelectorAll('.theme-dot').forEach(dot => {
            dot.addEventListener('click', function(e) {
                e.stopPropagation();
                setTheme(this.dataset.theme);
                // 保持展开状态，让用户看到切换效果
            });
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
