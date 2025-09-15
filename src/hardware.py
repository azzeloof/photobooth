import time
import threading
import logging
from typing import Optional, Dict, Callable, List, Any
from dataclasses import dataclass
from enum import Enum, auto
from contextlib import contextmanager

from common import PhotoboothError

# Try to import hardware-specific libraries. Fall back to mocks if unavailable.
try:
    import board  # type: ignore
    import digitalio  # type: ignore
    from adafruit_seesaw.seesaw import Seesaw  # type: ignore
    from adafruit_seesaw.digitalio import DigitalIO  # type: ignore
    from adafruit_seesaw.pwmout import PWMOut  # type: ignore
    SEESAW_AVAILABLE = True
except ImportError:
    SEESAW_AVAILABLE = False

    # Lightweight mocks to allow running without hardware
    class Seesaw:  # type: ignore
        pass

    class DigitalIO:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.value = True

    class PWMOut:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.duty_cycle = 0

    class _MockDirection:
        INPUT = 0

    class _MockPull:
        UP = 0

    class _MockDigitalIO:
        Direction = _MockDirection
        Pull = _MockPull

    digitalio = _MockDigitalIO()  # type: ignore
    board = object()  # type: ignore

logger = logging.getLogger(__name__)


class ButtonState(Enum):
    RELEASED = auto()
    PRESSED = auto()
    HELD = auto()


@dataclass
class ButtonEvent:
    """Event data for button interactions"""
    button_id: int
    state: ButtonState
    timestamp: float
    hold_duration: Optional[float] = None


@dataclass
class HardwareConfig:
    """Configuration for hardware components"""
    i2c_address: int = 0x3A
    poll_interval: float = 0.01  # 100Hz polling
    debounce_time: float = 0.05  # 50ms debounce
    hold_threshold: float = 1.0  # 1 second for hold detection
    led_brightness: float = 0.5  # 0.0 to 1.0

    # Button pin to LED pin mapping
    button_led_pins: Dict[int, int] = None

    def __post_init__(self):
        if self.button_led_pins is None:
            # Default pin mapping
            self.button_led_pins = {
                18: 12,
                19: 13,
                20: 0,
                2: 1
            }


class HardwareError(PhotoboothError):
    """Custom exception for hardware errors"""

    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(message, recoverable, component="Hardware")


class Button:
    """Enhanced button class with debouncing and state management"""

    def __init__(self, button_id: int, controller: Seesaw, button_pin: int,
                 led_pin: int, config: HardwareConfig):
        self.button_id = button_id
        self.controller = controller
        self.button_pin = button_pin
        self.led_pin = led_pin
        self.config = config

        # Button state tracking
        self.current_state = ButtonState.RELEASED
        self.last_state = ButtonState.RELEASED
        self.last_change_time = 0
        self.press_start_time = 0
        self.is_held = False

        # Callbacks
        self.press_callback: Optional[Callable[[], None]] = None
        self.release_callback: Optional[Callable[[], None]] = None
        self.hold_callback: Optional[Callable[[float], None]] = None

        # Hardware initialization
        self._initialize_hardware()

    def _initialize_hardware(self):
        """Initialize button and LED hardware"""
        try:
            # Configure button input
            self.btn = DigitalIO(self.controller, self.button_pin)
            self.btn.direction = digitalio.Direction.INPUT
            self.btn.pull = digitalio.Pull.UP

            # Configure LED output
            self.led = PWMOut(self.controller, self.led_pin)
            self.set_led_brightness(0)  # Start with LED off

            logger.debug(f"Button {self.button_id} initialized (pin {self.button_pin} -> LED {self.led_pin})")

        except Exception as error:
            raise HardwareError(f"Failed to initialize button {self.button_id}: {error}")

    def register_press_callback(self, callback: Callable[[], None]):
        """Register callback for button press events"""
        self.press_callback = callback

    def register_release_callback(self, callback: Callable[[], None]):
        """Register callback for button release events"""
        self.release_callback = callback

    def register_hold_callback(self, callback: Callable[[float], None]):
        """Register callback for button hold events (receives hold duration)"""
        self.hold_callback = callback

    def set_led_brightness(self, brightness: float):
        """Set LED brightness (0.0 to 1.0)"""
        try:
            # Convert to 16-bit PWM value
            pwm_value = int(brightness * 65535)
            self.led.duty_cycle = max(0, min(65535, pwm_value))
        except Exception as error:
            logger.error(f"Failed to set LED brightness for button {self.button_id}: {error}")

    def blink_led(self, duration: float = 0.2):
        """Blink the LED briefly"""
        try:
            original_brightness = self.led.duty_cycle / 65535.0
            self.set_led_brightness(self.config.led_brightness)
            time.sleep(duration)
            self.set_led_brightness(original_brightness)
        except Exception as error:
            logger.error(f"Failed to blink LED for button {self.button_id}: {error}")

    def poll(self) -> Optional[ButtonEvent]:
        """
        Poll button state and return event if the state changed

        Returns:
            ButtonEvent if state changed, None otherwise
        """
        try:
            current_time = time.time()

            # Read physical button state (inverted because of pull-up)
            physical_pressed = not self.btn.value

            # Debounce logic
            if physical_pressed != (self.current_state == ButtonState.PRESSED):
                if current_time - self.last_change_time > self.config.debounce_time:
                    self.last_change_time = current_time
                    self.last_state = self.current_state

                    if physical_pressed:
                        # Button pressed
                        self.current_state = ButtonState.PRESSED
                        self.press_start_time = current_time
                        self.is_held = False

                        # Trigger press callback
                        if self.press_callback:
                            try:
                                self.press_callback()
                            except Exception as error:
                                logger.error(f"Error in press callback for button {self.button_id}: {error}")

                        # Visual feedback
                        self.set_led_brightness(self.config.led_brightness)

                        return ButtonEvent(self.button_id, ButtonState.PRESSED, current_time)

                    else:
                        # Button released
                        hold_duration = current_time - self.press_start_time if self.press_start_time > 0 else 0
                        self.current_state = ButtonState.RELEASED

                        # Trigger release callback
                        if self.release_callback:
                            try:
                                self.release_callback()
                            except Exception as error:
                                logger.error(f"Error in release callback for button {self.button_id}: {error}")

                        # Turn off LED
                        self.set_led_brightness(0)

                        return ButtonEvent(self.button_id, ButtonState.RELEASED, current_time, hold_duration)

            # Check for hold detection
            elif (self.current_state == ButtonState.PRESSED and
                  not self.is_held and
                  current_time - self.press_start_time > self.config.hold_threshold):

                self.is_held = True
                hold_duration = current_time - self.press_start_time

                # Trigger hold callback
                if self.hold_callback:
                    try:
                        self.hold_callback(hold_duration)
                    except Exception as error:
                        logger.error(f"Error in hold callback for button {self.button_id}: {error}")

                # Visual feedback for hold (brighter LED)
                self.set_led_brightness(1.0)

                return ButtonEvent(self.button_id, ButtonState.HELD, current_time, hold_duration)

            return None

        except Exception as error:
            logger.error(f"Error polling button {self.button_id}: {error}")
            return None

    @property
    def is_pressed(self) -> bool:
        """Check if the button is currently pressed"""
        return self.current_state in [ButtonState.PRESSED, ButtonState.HELD]


class Hardware:
    """Enhanced hardware manager with robust error handling and threading"""

    def __init__(self, config: Optional[HardwareConfig] = None):
        self.config = config or HardwareConfig()

        # Hardware state
        self.seesaw: Optional[Seesaw] = None
        self.buttons: List[Button] = []
        self.running = False
        self._initialized = False

        # Threading
        self.hardware_thread: Optional[threading.Thread] = None
        self.event_callbacks: Dict[int, List[Callable[[ButtonEvent], None]]] = {}
        self.global_callbacks: List[Callable[[ButtonEvent], None]] = []

        # Initialize hardware
        self._initialize_hardware()

    def _initialize_hardware(self):
        """Initialize the Seesaw controller and buttons"""
        if not SEESAW_AVAILABLE:
            raise HardwareError("Seesaw library not available. Install with: pip install adafruit-circuitpython-seesaw")

        try:
            # Initialize I2C and Seesaw
            i2c = board.I2C()
            self.seesaw = Seesaw(i2c, addr=self.config.i2c_address)

            logger.info(f"Seesaw controller initialized at address 0x{self.config.i2c_address:02X}")

            # Initialize buttons
            for button_id, (button_pin, led_pin) in enumerate(self.config.button_led_pins.items()):
                button = Button(
                    button_id=button_id,
                    controller=self.seesaw,
                    button_pin=button_pin,
                    led_pin=led_pin,
                    config=self.config
                )
                self.buttons.append(button)
                self.event_callbacks[button_id] = []

            self._initialized = True
            logger.info(f"Hardware initialized with {len(self.buttons)} buttons")

        except Exception as error:
            raise HardwareError(f"Hardware initialization failed: {error}", recoverable=False)

    def register_callback(self, button_id: int, callback: Callable[[], None]):
        """Register a simple callback for button press (backwards compatibility)"""
        if button_id >= len(self.buttons):
            raise HardwareError(f"Invalid button ID: {button_id}")

        # Wrap the simple callback to work with the new event system
        def event_wrapper():
            try:
                callback()
            except Exception as error:
                logger.error(f"Error in callback for button {button_id}: {error}")

        self.buttons[button_id].register_press_callback(event_wrapper)

    def register_event_callback(self, button_id: int, callback: Callable[[ButtonEvent], None]):
        """Register an event-based callback for a specific button"""
        if button_id >= len(self.buttons):
            raise HardwareError(f"Invalid button ID: {button_id}")

        self.event_callbacks[button_id].append(callback)

    def register_global_callback(self, callback: Callable[[ButtonEvent], None]):
        """Register a callback that receives all button events"""
        self.global_callbacks.append(callback)

    def set_button_led(self, button_id: int, brightness: float):
        """Set LED brightness for a specific button"""
        if button_id >= len(self.buttons):
            raise HardwareError(f"Invalid button ID: {button_id}")

        self.buttons[button_id].set_led_brightness(brightness)

    def blink_button_led(self, button_id: int, duration: float = 0.2):
        """Blink LED for a specific button"""
        if button_id >= len(self.buttons):
            raise HardwareError(f"Invalid button ID: {button_id}")

        # Run blink in a separate thread to avoid blocking
        threading.Thread(
            target=self.buttons[button_id].blink_led,
            args=(duration,),
            daemon=True
        ).start()

    def get_button_state(self, button_id: int) -> ButtonState:
        """Get the current state of a button"""
        if button_id >= len(self.buttons):
            raise HardwareError(f"Invalid button ID: {button_id}")

        return self.buttons[button_id].current_state

    def is_button_pressed(self, button_id: int) -> bool:
        """Check if a specific button is pressed"""
        if button_id >= len(self.buttons):
            return False

        return self.buttons[button_id].is_pressed

    def _poll_buttons(self):
        """Poll all buttons and handle events"""
        for button in self.buttons:
            event = button.poll()
            if event:
                # Call button-specific callbacks
                for callback in self.event_callbacks.get(event.button_id, []):
                    try:
                        callback(event)
                    except Exception as error:
                        logger.error(f"Error in event callback for button {event.button_id}: {error}")

                # Call global callbacks
                for callback in self.global_callbacks:
                    try:
                        callback(event)
                    except Exception as error:
                        logger.error(f"Error in global event callback: {error}")

    def run(self):
        """Main hardware polling loop"""
        if not self._initialized:
            raise HardwareError("Hardware not initialized")

        self.running = True
        logger.info("Starting hardware polling loop")

        try:
            while self.running:
                self._poll_buttons()
                time.sleep(self.config.poll_interval)
        except Exception as error:
            logger.error(f"Hardware polling loop error: {error}")
        finally:
            logger.info("Hardware polling loop stopped")

    def start(self):
        """Start hardware polling in a separate thread"""
        if self.hardware_thread and self.hardware_thread.is_alive():
            logger.warning("Hardware thread already running")
            return

        self.hardware_thread = threading.Thread(target=self.run, daemon=True)
        self.hardware_thread.start()
        logger.info("Hardware thread started")

    def stop(self):
        """Stop hardware polling"""
        logger.info("Stopping hardware...")
        self.running = False

        # Turn off all LEDs
        for button in self.buttons:
            try:
                button.set_led_brightness(0)
            except Exception as error:
                logger.error(f"Error turning off LED for button {button.button_id}: {error}")

        # Wait for the thread to finish
        if self.hardware_thread and self.hardware_thread.is_alive():
            self.hardware_thread.join(timeout=1.0)

    def test_all_buttons(self):
        """Test all buttons and LEDs"""
        logger.info("Testing all buttons and LEDs...")

        for button in self.buttons:
            logger.info(f"Testing button {button.button_id}...")

            # Blink LED
            button.blink_led(0.3)
            time.sleep(0.5)

            # Test button state
            logger.info(f"Button {button.button_id} pressed: {button.is_pressed}")

    @property
    def is_initialized(self) -> bool:
        """Check if hardware is initialized"""
        return self._initialized

    @property
    def button_count(self) -> int:
        """Get the number of available buttons"""
        return len(self.buttons)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


    def on_button_press():
        print("Button pressed!")


    def on_button_event(event: ButtonEvent):
        print(f"Button {event.button_id}: {event.state.name} at {event.timestamp}")
        if event.hold_duration:
            print(f"  Hold duration: {event.hold_duration:.2f}s")


    try:
        config = HardwareConfig(poll_interval=0.01)

        with Hardware(config) as hw:
            # Register callbacks (both old and new style)
            hw.register_callback(0, on_button_press)
            hw.register_global_callback(on_button_event)

            # Test hardware
            hw.test_all_buttons()

            # Start polling
            hw.start()

            print("Hardware running. Press buttons or Ctrl+C to exit...")
            try:
                while True:
                    time.sleep(1)
                    # You could add status reporting here
                    pressed_buttons = [i for i in range(hw.button_count) if hw.is_button_pressed(i)]
                    if pressed_buttons:
                        print(f"Currently pressed: {pressed_buttons}")
            except KeyboardInterrupt:
                print("Exiting...")

    except HardwareError as e:
        print(f"Hardware error: {e}")
        if not e.recoverable:
            print("This error is not recoverable. Check hardware connections.")
    except Exception as e:
        print(f"Unexpected error: {e}")