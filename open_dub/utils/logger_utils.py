import logging
import time
import sys
import random
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

class CatLogger:
    CAT_FACES: Dict[str, str] = {
        'debug': '(= ФェФ=)',
        'info': '(=^･ω･^=)',
        'warning': '(=⊙_⊙=)',
        'error': '(=｀ェ´=)',
        'critical': '(=ｘェｘ=)'
    }
    
    CAT_ANIMATIONS: Dict[str, List[str]] = {
        'thinking': ['(=･ω･=)', '(=｡･ω･｡=)', '(=･ω･=)～', '(=｡･ω･｡=)～'],
        'loading': ['(=｡･ω･｡=)', '(=｡･ω･｡=)_', '(=｡･ω･｡=)__', '(=｡･ω･｡=)___'],
        'running': ['(=^･ω･^=)ﾉ', '(=^･ω･^=)/', '(=^･ω･^=)～', '(=^･ω･^=)~~'],
        'sleeping': ['(=^･ｪ･^=).｡｡', '(=^･ｚ･^=).｡｡', '(=^･ω･^=).｡｡', '(=^･ｚｚ･^=).｡｡']
    }
    
    COLORS: Dict[str, str] = {
        'debug': '\033[36m',    # Cyan
        'info': '\033[32m',     # Green
        'warning': '\033[33m',  # Yellow
        'error': '\033[31m',    # Red
        'critical': '\033[35m', # Purple
        'reset': '\033[0m'      # Reset
    }
    
    def __init__(self, name: str = "CatLogger", level: int = logging.INFO, 
                 use_colors: bool = True, show_animations: bool = True):
        self.name: str = name
        self.level: int = level
        self.use_colors: bool = use_colors
        self.show_animations: bool = show_animations
        
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            self.logger.addHandler(console_handler)
    
    def _format_message(self, level: str, message: str) -> str:
        timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cat_face: str = self.CAT_FACES.get(level, '(=^･ω･^=)')
        
        if self.use_colors:
            color_code: str = self.COLORS.get(level, self.COLORS['reset'])
            return f"{color_code}[{timestamp}] {cat_face} {message}{self.COLORS['reset']}"
        else:
            return f"[{timestamp}] {cat_face} {message}"
    
    def _log(self, level: str, message: str, *args, **kwargs) -> None:
        if args:
            message = message % args
            
        formatted_message: str = self._format_message(level, message)
        getattr(self.logger, level)(formatted_message, **kwargs)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        self._log('debug', message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        self._log('info', message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        self._log('warning', message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        self._log('error', message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        self._log('critical', message, *args, **kwargs)
    
    def animate(self, animation_type: str = 'thinking', duration: float = 2.0, 
                message: Optional[str] = None) -> None:
        if not self.show_animations:
            return
            
        frames: List[str] = self.CAT_ANIMATIONS.get(animation_type, self.CAT_ANIMATIONS['thinking'])
        start_time: float = time.time()
        
        try:
            while time.time() - start_time < duration:
                for frame in frames:
                    sys.stdout.write(f"\r{frame} {message if message else ''}   ")
                    sys.stdout.flush()
                    time.sleep(0.2)
                    
            sys.stdout.write("\r" + " " * (len(frames[0]) + (len(message) if message else 0) + 3) + "\r")
            sys.stdout.flush()
            
        except KeyboardInterrupt:
            sys.stdout.write("\r" + " " * (len(frames[0]) + (len(message) if message else 0) + 3) + "\r")
            sys.stdout.flush()
    
    def cat_say(self, message: str) -> None:
        lines: List[str] = message.split('\n')
        max_length: int = max(len(line) for line in lines)
        
        print(" " + "_" * (max_length + 2))
        for line in lines:
            print(f"< {line.ljust(max_length)} >")
        print(" " + "-" * (max_length + 2))
        
        print("  \\/")
        print(" (=^･ω･^=)")
        print(" /|  |\\")
        print("  U  U")
    
    def progress_bar(self, total: int, message: str = "Processing", 
                    bar_length: int = 30) -> None:
        if not self.show_animations:
            return
            
        cat_emojis: List[str] = ['(=･ω･=)', '(=^･ω･^=)', '(=｡･ω･｡=)']
        
        for i in range(total + 1):
            percent: float = i * 100.0 / total
            progress: int = int(bar_length * i / total)
            bar: str = '▓' * progress + '░' * (bar_length - progress)
            cat_pos: int = min(progress, bar_length - 1)
            
            bar_with_cat: str = bar[:cat_pos] + random.choice(cat_emojis) + bar[cat_pos+1:]
            
            sys.stdout.write(f"\r{message}: [{bar_with_cat}] {percent:.1f}%")
            sys.stdout.flush()
            time.sleep(0.05)
            
        print()


def get_cat_logger(name: str = "CatLogger", level: str = "INFO", 
                  use_colors: bool = True, show_animations: bool = True) -> CatLogger:
    level_map: Dict[str, int] = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level: int = level_map.get(level.upper(), logging.INFO)
    
    return CatLogger(name=name, level=log_level, use_colors=use_colors, show_animations=show_animations)