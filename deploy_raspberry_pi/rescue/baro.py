"""
BMP388 气压传感器（I2C）读取与相对高度计算

补偿公式取自 Bosch BMP388 数据手册的浮点实现。parse_calibration / compensate 只依赖标准库，
可在无硬件时测试；BMP388 类通过 smbus2 访问 I2C 总线（树莓派上: pip install smbus2，
并在 raspi-config 中启用 I2C）。DFRobot Gravity BMP388 模块默认地址 0x77（SDO 接地时为 0x76）。
"""

import struct
import time
from dataclasses import dataclass

from .geo_mapping import altitude_from_pressure

CHIP_IDS = (0x50, 0x60)   # BMP388 / BMP390，寄存器与补偿公式相同
REG_CHIP_ID = 0x00
REG_DATA = 0x04           # PRESS_XLSB … TEMP_MSB，共 6 字节
REG_PWR_CTRL = 0x1B
REG_OSR = 0x1C
REG_ODR = 0x1D
REG_CONFIG = 0x1F
REG_CALIB = 0x31          # NVM_PAR_T1 起，共 21 字节
REG_CMD = 0x7E
CMD_SOFT_RESET = 0xB6


@dataclass
class Calibration:
    t1: float
    t2: float
    t3: float
    p1: float
    p2: float
    p3: float
    p4: float
    p5: float
    p6: float
    p7: float
    p8: float
    p9: float
    p10: float
    p11: float


def parse_calibration(raw: bytes) -> Calibration:
    """21 字节 NVM 校准数据 -> 浮点系数（数据手册 9.1 节）"""
    t1, t2, t3, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = struct.unpack("<HHbhhbbHHbbhbb", bytes(raw))
    return Calibration(
        t1=t1 * 2.0 ** 8, t2=t2 / 2.0 ** 30, t3=t3 / 2.0 ** 48,
        p1=(p1 - 2 ** 14) / 2.0 ** 20, p2=(p2 - 2 ** 14) / 2.0 ** 29,
        p3=p3 / 2.0 ** 32, p4=p4 / 2.0 ** 37, p5=p5 * 2.0 ** 3, p6=p6 / 2.0 ** 6,
        p7=p7 / 2.0 ** 8, p8=p8 / 2.0 ** 15, p9=p9 / 2.0 ** 48, p10=p10 / 2.0 ** 48, p11=p11 / 2.0 ** 65,
    )


def compensate(raw_press: int, raw_temp: int, c: Calibration) -> tuple[float, float]:
    """原始 ADC 值 -> (气压 hPa, 温度 °C)"""
    pd1 = raw_temp - c.t1
    t = pd1 * c.t2 + pd1 * pd1 * c.t3
    out1 = c.p5 + c.p6 * t + c.p7 * t ** 2 + c.p8 * t ** 3
    out2 = raw_press * (c.p1 + c.p2 * t + c.p3 * t ** 2 + c.p4 * t ** 3)
    out3 = raw_press ** 2 * (c.p9 + c.p10 * t) + raw_press ** 3 * c.p11
    return (out1 + out2 + out3) / 100.0, t


def unpack_data(data: bytes) -> tuple[int, int]:
    """6 字节数据寄存器 -> (原始气压, 原始温度)，均为 24 位小端"""
    d = bytes(data)
    return d[0] | d[1] << 8 | d[2] << 16, d[3] | d[4] << 8 | d[5] << 16


class BMP388:
    """按数据手册“无人机”推荐配置：气压 ×8 / 温度 ×1 过采样，IIR 系数 3，输出 50 Hz"""

    def __init__(self, bus: int = 1, address: int = 0x77, smbus=None):
        if smbus is None:
            from smbus2 import SMBus
            smbus = SMBus(bus)
        self.bus, self.addr = smbus, address
        chip = self.bus.read_byte_data(self.addr, REG_CHIP_ID)
        if chip not in CHIP_IDS:
            raise RuntimeError(f"I2C 地址 0x{address:02X} 处芯片 ID 为 0x{chip:02X}，不是 BMP388/BMP390")
        self.bus.write_byte_data(self.addr, REG_CMD, CMD_SOFT_RESET)
        time.sleep(0.01)
        self.calib = parse_calibration(self.bus.read_i2c_block_data(self.addr, REG_CALIB, 21))
        self.bus.write_byte_data(self.addr, REG_OSR, 0x03)       # osr_p=×8, osr_t=×1
        self.bus.write_byte_data(self.addr, REG_ODR, 0x02)       # 200/2^2 = 50 Hz
        self.bus.write_byte_data(self.addr, REG_CONFIG, 0x02 << 1)  # IIR 系数 3
        self.bus.write_byte_data(self.addr, REG_PWR_CTRL, 0x33)  # 气压+温度使能，normal 模式
        time.sleep(0.05)

    def read(self) -> tuple[float, float]:
        """返回 (气压 hPa, 温度 °C)"""
        raw_p, raw_t = unpack_data(self.bus.read_i2c_block_data(self.addr, REG_DATA, 6))
        return compensate(raw_p, raw_t, self.calib)


class RelativeAltimeter:
    """起飞前在地面采样求参考气压，之后输出相对地面的高度（米）"""

    def __init__(self, sensor, samples: int = 50, interval_s: float = 0.02):
        readings = []
        for _ in range(samples):
            readings.append(sensor.read())
            time.sleep(interval_s)
        self.sensor = sensor
        self.ground_hpa = sum(p for p, _ in readings) / len(readings)

    def altitude(self) -> tuple[float, float, float]:
        """返回 (相对高度 m, 气压 hPa, 温度 °C)"""
        p, t = self.sensor.read()
        return altitude_from_pressure(p, self.ground_hpa, t), p, t
