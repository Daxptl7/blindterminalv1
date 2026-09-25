import unittest
from pico_firmware.main import ButtonController


class FirmwareTests(unittest.TestCase):
    def setUp(self):
        self.messages = []
        self.controller = ButtonController(self.messages.append, lambda a, b: a-b)
        self.now = 0
        self.values = [0, 0, 0]

    def advance(self, milliseconds):
        for _ in range(milliseconds // 5):
            self.now += 5
            self.controller.update(self.values, self.now)

    def press(self, index, duration=60):
        self.values[index] = 1
        self.advance(duration)
        self.values[index] = 0
        self.advance(60)

    def test_single_confirm_becomes_space(self):
        self.press(2)
        self.advance(550)
        self.assertEqual(self.messages, ['RAW:3', 'WORD_SPACE'])

    def test_double_press_is_confirm_without_space(self):
        self.press(2)
        self.press(2)
        self.advance(550)
        self.assertEqual(self.messages, ['RAW:3', 'RAW:3', 'CONFIRM'])

    def test_triple_press_preserves_all_raw_events(self):
        for _ in range(3):
            self.press(2)
        self.assertEqual(self.messages.count('RAW:3'), 3)

    def test_hold_emits_one_backspace_no_space(self):
        self.press(2, 1000)
        self.advance(600)
        self.assertEqual(self.messages, ['RAW:3', 'BACKSPACE'])

    def test_other_buttons_work_during_hold(self):
        self.values[0] = 1
        self.advance(100)
        self.press(1)
        self.press(2)
        self.assertEqual(self.messages[:3], ['RAW:1', 'RAW:2', 'RAW:3'])

    def test_bounce_is_ignored(self):
        self.values[0] = 1
        self.advance(10)
        self.values[0] = 0
        self.advance(100)
        self.assertEqual(self.messages, [])

    def test_morse_letter_before_confirm_is_not_lost(self):
        self.press(0)
        self.press(1)
        self.press(2)
        self.press(2)
        self.assertEqual(self.messages[-2:], ['LETTER:A', 'CONFIRM'])

    def test_letter_timeout(self):
        self.press(0)
        self.advance(1500)
        self.assertEqual(self.messages, ['RAW:1', 'LETTER:E'])

    def test_single_presses_outside_window_do_not_confirm(self):
        self.press(2)
        self.advance(600)
        self.press(2)
        self.advance(600)
        self.assertEqual(self.messages, ['RAW:3', 'WORD_SPACE'] * 2)

    def test_ticks_wrap(self):
        self.controller.diff = lambda a, b: ((a-b+2048) % 4096)-2048
        self.controller.update([1, 0, 0], 4080)
        self.controller.update([1, 0, 0], 20)
        self.assertEqual(self.messages, ['RAW:1'])


class MenuDigitTests(unittest.TestCase):
    def read(self, events):
        from unittest import mock
        from modules import morse_serial
        clock = [0.0]
        bridge = morse_serial.MorseSerial.__new__(morse_serial.MorseSerial)
        iterator = iter(events)
        def get_message(timeout=None):
            event = next(iterator, None)
            if event is None:
                return None
            clock[0], message = event
            return message
        bridge.get_message = get_message
        with mock.patch.object(morse_serial.time, 'monotonic', side_effect=lambda: clock[0]):
            return bridge.read_menu_digit(timeout=120)

    def test_slow_ocr_sequence_ignores_interleaved_letters(self):
        self.assertEqual(self.read([(0, 'RAW:1'), (1.5, 'LETTER:E'),
            (2, 'RAW:2'), (3.5, 'LETTER:T'), (4, 'RAW:2'),
            (5.5, 'LETTER:T'), (6, 'RAW:2'), (7.5, 'LETTER:T'),
            (8, 'RAW:2')]), '1')

    def test_all_ten_digits(self):
        for digit, symbols in enumerate(('-----', '.----', '..---', '...--',
                                        '....-', '.....', '-....', '--...', '---..', '----.')):
            with self.subTest(digit=digit):
                self.assertEqual(self.read([(i*.2, 'RAW:1' if s == '.' else 'RAW:2')
                                            for i, s in enumerate(symbols)]), str(digit))

    def test_long_pause_resets_partial_digit(self):
        self.assertIsNone(self.read([(0, 'RAW:1'), (8, 'RAW:2'),
                                    (9, 'RAW:2'), (10, 'RAW:2'), (11, 'RAW:2')]))

    def test_legacy_decoded_digit(self):
        self.assertEqual(self.read([(0, 'LETTER:1')]), '1')

    def test_partial_digit_then_reboot_is_discarded(self):
        self.assertIsNone(self.read([(0, 'RAW:1'), (1, 'READY'),
                       (2, 'RAW:2'), (3, 'RAW:2'), (4, 'RAW:2'), (5, 'RAW:2')]))


if __name__ == '__main__':
    unittest.main()
