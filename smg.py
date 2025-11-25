from gpiozero import LED, DigitalOutputDevice
from time import sleep

# GPIO pin definitions
segment_pins = [16, 21, 19, 6, 5, 20, 26, 13]  # a, b, c, d, e, f, g, dp
digit_pins = [18, 23, 24, 22]                 # Common anode pins for 4 digits

# Initialize GPIO pins
segments = [LED(pin) for pin in segment_pins]  # Segments controlled by LED class
digits = [DigitalOutputDevice(pin, active_high=True) for pin in digit_pins]  # Digits controlled by DigitalOutputDevice

# Segment codes (common anode, active low for segments)
segment_codes = [
    0xc0,  # 0
    0xf9,  # 1
    0xa4,  # 2
    0xb0,  # 3
    0x99,  # 4
    0x92,  # 5
    0x82,  # 6
    0xf8,  # 7
    0x80,  # 8
    0x90   # 9
]

# Display a single digit by controlling anode (active low)
def display_digit(digit, value):
    # Turn off all digits (ensure only one digit is active)
    for d in digits:
        d.off()
    
    # Set segment values (active low logic)
    for i, segment in enumerate(segments):
        segment.value = ((segment_codes[value] >> i) & 1)  # Active low
    
    # Enable the selected digit
    digits[digit].on()

# Display a multi-digit number using dynamic scanning
def display_number(number):
    numbers = [int(d) for d in str(number).zfill(4)]  # Ensure 4 digits with leading zeros
    for i in range(4):  # Loop through each digit
        display_digit(i, numbers[i])  # Display each digit
        sleep(0.002)  # Delay to control refresh rate

# Startup animation for the display
def startup_animation():
    for _ in range(3):  # Flash all segments and digits 3 times
        for segment in segments:
            segment.on()
        for digit in digits:
            digit.on()
        sleep(0.2)
        for segment in segments:
            segment.off()
        for digit in digits:
            digit.off()
        sleep(0.2)

if __name__ == "__main__":
# Main loop
    try:
        startup_animation()
        while True:
            display_number(13)  # Replace with the number to display
    except KeyboardInterrupt:
        # Turn off all segments and digits on exit
        for segment in segments:
            segment.off()
        for digit in digits:
            digit.off()

