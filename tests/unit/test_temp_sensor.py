# Unit tests for chamber_mpc.temp_sensor
import pytest
from chamber_mpc.temp_sensor import ChamberMpcTempSensor, REPORT_TIME


class DummyReactor:
    NOW = 0.0

    def __init__(self):
        self.t = 10.0
        self.timer_cb = None

    def monotonic(self):
        return self.t

    def register_timer(self, cb, waketime):
        self.timer_cb = cb
        return 'timer'


class DummyConfig:
    def __init__(self, printer, values=None):
        self._printer = printer
        self._values = values or {}

    def get_printer(self):
        return self._printer

    def get(self, key, default=None):
        return self._values.get(key, default)


class DummyModel:
    def __init__(self, temp):
        self.state_chamber_temp = temp


class DummyControl:
    def __init__(self, model=None):
        self.model = model


class DummyHeater:
    def __init__(self, raw_temp=30.0, control=None):
        self.raw_temp = raw_temp
        self.control = control

    def get_temp(self, eventtime):
        return (self.raw_temp, 0.0)


class DummyHeaters:
    def __init__(self, heaters):
        self._heaters = heaters

    def lookup_heater(self, name):
        return self._heaters[name]


class DummyMpcModule:
    def __init__(self, heater_name):
        self.heater_name = heater_name


class DummyPrinter:
    def __init__(self, heater, heater_name='chamber'):
        self.reactor = DummyReactor()
        self.handlers = {}
        self.objects = {
            'heaters': DummyHeaters({heater_name: heater}),
            'chamber_mpc': DummyMpcModule(heater_name),
        }

    def get_reactor(self):
        return self.reactor

    def register_event_handler(self, name, fn):
        self.handlers[name] = fn

    def lookup_object(self, name, default=None):
        return self.objects.get(name, default)


class TestChamberMpcTempSensor:
    def test_reports_estimated_chamber_temp_when_mpc_active(self):
        model = DummyModel(64.8)
        heater = DummyHeater(raw_temp=62.1, control=DummyControl(model))
        printer = DummyPrinter(heater)
        sensor = ChamberMpcTempSensor(DummyConfig(printer))
        sensor.setup_minmax(-10.0, 100.0)

        reported = []
        sensor.setup_callback(lambda t, temp: reported.append((t, temp)))
        assert sensor.get_report_time_delta() == pytest.approx(REPORT_TIME)

        printer.handlers['klippy:ready']()
        next_t = sensor._sample_temp(10.0)
        assert next_t == pytest.approx(10.0 + REPORT_TIME)
        assert reported == [(10.0, 64.8)]
        assert sensor.get_temp(10.0) == (64.8, 0.0)
        assert sensor.get_status(10.0) == {'temperature': 64.8}

    def test_falls_back_to_raw_heater_temp_when_no_model(self):
        heater = DummyHeater(raw_temp=41.5, control=DummyControl(model=None))
        printer = DummyPrinter(heater)
        sensor = ChamberMpcTempSensor(DummyConfig(printer))

        reported = []
        sensor.setup_callback(lambda t, temp: reported.append((t, temp)))
        printer.handlers['klippy:ready']()
        sensor._sample_temp(10.0)
        assert reported == [(10.0, 41.5)]

    def test_explicit_heater_name_overrides_default(self):
        model = DummyModel(72.3)
        heater = DummyHeater(raw_temp=70.0, control=DummyControl(model))
        printer = DummyPrinter(heater, heater_name='oven')
        sensor = ChamberMpcTempSensor(
            DummyConfig(printer, {'heater_name': 'oven'}))

        reported = []
        sensor.setup_callback(lambda t, temp: reported.append((t, temp)))
        printer.handlers['klippy:ready']()
        sensor._sample_temp(10.0)
        assert reported == [(10.0, 72.3)]
