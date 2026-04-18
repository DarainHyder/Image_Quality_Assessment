import React, { useEffect, useRef } from 'react';

const CursorTrail = () => {
    const canvasRef = useRef(null);
    const mouse = useRef({ x: 0, y: 0 });
    const dots = useRef([]);
    const animationFrame = useRef(null);

    const DOT_COUNT = 25;
    const DOT_LIFESPAN = 20;

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };

        const onMouseMove = (e) => {
            mouse.current = { x: e.clientX, y: e.clientY };
            dots.current.push({
                x: mouse.current.x,
                y: mouse.current.y,
                life: DOT_LIFESPAN,
                size: Math.random() * 3 + 1,
                color: `hsla(${230 + Math.random() * 20}, 70%, 60%, `
            });
        };

        const render = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            dots.current.forEach((dot, index) => {
                dot.life--;
                if (dot.life <= 0) {
                    dots.current.splice(index, 1);
                    return;
                }

                // Draw trail segment or dot
                const opacity = dot.life / DOT_LIFESPAN;
                ctx.beginPath();
                ctx.arc(dot.x, dot.y, dot.size * opacity, 0, Math.PI * 2);
                ctx.fillStyle = dot.color + opacity + ')';
                ctx.fill();

                // Move slightly
                dot.y += 0.5;
            });

            animationFrame.current = requestAnimationFrame(render);
        };

        window.addEventListener('resize', resize);
        window.addEventListener('mousemove', onMouseMove);
        resize();
        render();

        return () => {
            window.removeEventListener('resize', resize);
            window.removeEventListener('mousemove', onMouseMove);
            cancelAnimationFrame(animationFrame.current);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="fixed inset-0 pointer-events-none z-[9999]"
            style={{ mixBlendMode: 'screen' }}
        />
    );
};

export default CursorTrail;
