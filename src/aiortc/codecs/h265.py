import fractions
import logging
import math
from itertools import tee
from struct import pack, unpack_from
from typing import Iterator, List, Optional, Sequence, Tuple, Type, TypeVar

import av
from av.frame import Frame
from av.packet import Packet

from ..jitterbuffer import JitterFrame
from ..mediastreams import VIDEO_TIME_BASE, convert_timebase
from .base import Decoder, Encoder

logger = logging.getLogger(__name__)

DEFAULT_BITRATE = 1500000  # 1.5 Mbps
MIN_BITRATE = 500000  # 500 kbps
MAX_BITRATE = 5000000  # 5 Mbps

MAX_FRAME_RATE = 30
PACKET_MAX = 1280

NAL_TYPE_FU = 49
NAL_TYPE_AP = 48

NAL_HEADER_SIZE = 2
FU_HEADER_SIZE = 3
LENGTH_FIELD_SIZE = 2
AP_HEADER_SIZE = NAL_HEADER_SIZE + LENGTH_FIELD_SIZE

DESCRIPTOR_T = TypeVar("DESCRIPTOR_T", bound="H265PayloadDescriptor")
T = TypeVar("T")


def pairwise(iterable: Sequence[T]) -> Iterator[Tuple[T, T]]:
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class H265PayloadDescriptor:
    def __init__(self, first_fragment):
        self.first_fragment = first_fragment

    def __repr__(self):
        return f"H265PayloadDescriptor(FF={self.first_fragment})"

    @classmethod
    def parse(cls: Type[DESCRIPTOR_T], data: bytes) -> Tuple[DESCRIPTOR_T, bytes]:
        output = bytes()

        # NAL unit header
        if len(data) < 2:
            raise ValueError("NAL unit is too short")
        nal_type = (data[0] >> 1) & 0x3F
        layer_id = ((data[0] & 0x01) << 5) | ((data[1] >> 3) & 0x1F)
        tid = data[1] & 0x07
        pos = NAL_HEADER_SIZE

        if nal_type in range(0, 48):
            # single NAL unit
            output = bytes([0, 0, 0, 1]) + data
            obj = cls(first_fragment=True)
        elif nal_type == NAL_TYPE_FU:
            # fragmentation unit
            if len(data) < 3:
                raise ValueError("FU is too short")
            fu_header = data[pos]
            original_nal_type = fu_header & 0x3F
            first_fragment = bool(fu_header & 0x80)
            pos += 1

            if first_fragment:
                original_nal_header = bytes(
                    [
                        (original_nal_type << 1) | (layer_id >> 5),
                        ((layer_id & 0x1F) << 3) | tid,
                    ]
                )
                output += bytes([0, 0, 0, 1])
                output += original_nal_header
            output += data[pos:]

            obj = cls(first_fragment=first_fragment)
        elif nal_type == NAL_TYPE_AP:
            # aggregation packet
            offsets = []
            while pos < len(data):
                if len(data) < pos + LENGTH_FIELD_SIZE:
                    raise ValueError("AP length field is truncated")
                nalu_size = unpack_from("!H", data, pos)[0]
                pos += LENGTH_FIELD_SIZE
                offsets.append(pos)

                pos += nalu_size
                if len(data) < pos:
                    raise ValueError("AP data is truncated")

            offsets.append(len(data) + LENGTH_FIELD_SIZE)
            for start, end in pairwise(offsets):
                end -= LENGTH_FIELD_SIZE
                output += bytes([0, 0, 0, 1])
                output += data[start:end]

            obj = cls(first_fragment=True)
        else:
            raise ValueError(f"NAL unit type {nal_type} is not supported")

        return obj, output


def create_encoder_context(
    codec_name: str, width: int, height: int, bitrate: int
) -> Tuple[av.CodecContext, bool]:
    codec = av.CodecContext.create(codec_name, "w")
    codec.width = width
    codec.height = height
    codec.bit_rate = bitrate
    codec.pix_fmt = "yuv420p"
    codec.framerate = fractions.Fraction(MAX_FRAME_RATE, 1)
    codec.time_base = fractions.Fraction(1, MAX_FRAME_RATE)
    codec.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
    }
    codec.open()
    return codec, codec_name == "hevc_omx"


class H265Encoder(Encoder):
    def __init__(self) -> None:
        self.buffer_data = b""
        self.buffer_pts: Optional[int] = None
        self.codec: Optional[av.CodecContext] = None
        self.codec_buffering = False
        self.__target_bitrate = DEFAULT_BITRATE

    @staticmethod
    def _packetize_fu(data: bytes) -> List[bytes]:
        available_size = PACKET_MAX - FU_HEADER_SIZE
        payload_size = len(data) - NAL_HEADER_SIZE
        num_packets = math.ceil(payload_size / available_size)
        num_larger_packets = payload_size % num_packets
        package_size = payload_size // num_packets

        nal_header = data[:NAL_HEADER_SIZE]
        nal_type = (nal_header[0] >> 1) & 0x3F
        layer_id = ((nal_header[0] & 0x01) << 5) | ((nal_header[1] >> 3) & 0x1F)
        tid = nal_header[1] & 0x07

        fu_indicator = bytes(
            [(NAL_TYPE_FU << 1) | (layer_id >> 5), ((layer_id & 0x1F) << 3) | tid]
        )

        fu_header_end = bytes([0x40 | nal_type])
        fu_header_middle = bytes([nal_type])
        fu_header_start = bytes([0x80 | nal_type])
        fu_header = fu_header_start

        packages = []
        offset = NAL_HEADER_SIZE
        while offset < len(data):
            if num_larger_packets > 0:
                num_larger_packets -= 1
                payload = data[offset : offset + package_size + 1]
                offset += package_size + 1
            else:
                payload = data[offset : offset + package_size]
                offset += package_size

            if offset == len(data):
                fu_header = fu_header_end

            packages.append(fu_indicator + fu_header + payload)

            fu_header = fu_header_middle
        assert offset == len(data), "incorrect fragment data"

        return packages

    @staticmethod
    def _packetize_ap(
        data: bytes, packages_iterator: Iterator[bytes]
    ) -> Tuple[bytes, bytes]:
        counter = 0
        available_size = PACKET_MAX - AP_HEADER_SIZE

        ap_header = bytes([(NAL_TYPE_AP << 1), 0x01])  # Assuming layer_id=0, tid=1

        payload = bytes()
        try:
            nalu = data  # with header
            while len(nalu) <= available_size and counter < 9:
                available_size -= LENGTH_FIELD_SIZE + len(nalu)
                counter += 1
                payload += pack("!H", len(nalu)) + nalu
                nalu = next(packages_iterator)

            if counter == 0:
                nalu = next(packages_iterator)
        except StopIteration:
            nalu = None

        if counter <= 1:
            return data, nalu
        else:
            return ap_header + payload, nalu

    @staticmethod
    def _split_bitstream(buf: bytes) -> Iterator[bytes]:
        i = 0
        while True:
            i = buf.find(b"\x00\x00\x00\x01", i)
            if i == -1:
                return

            i += 4
            nal_start = i

            i = buf.find(b"\x00\x00\x00\x01", i)
            if i == -1:
                yield buf[nal_start : len(buf)]
                return
            else:
                yield buf[nal_start:i]

    @classmethod
    def _packetize(cls, packages: Iterator[bytes]) -> List[bytes]:
        packetized_packages = []

        packages_iterator = iter(packages)
        package = next(packages_iterator, None)
        while package is not None:
            if len(package) > PACKET_MAX:
                packetized_packages.extend(cls._packetize_fu(package))
                package = next(packages_iterator, None)
            else:
                packetized, package = cls._packetize_ap(package, packages_iterator)
                packetized_packages.append(packetized)

        return packetized_packages

    def _encode_frame(
        self, frame: av.VideoFrame, force_keyframe: bool
    ) -> Iterator[bytes]:
        if self.codec and (
            frame.width != self.codec.width
            or frame.height != self.codec.height
            or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate
            > 0.1
        ):
            self.buffer_data = b""
            self.buffer_pts = None
            self.codec = None

        if force_keyframe:
            frame.pict_type = av.video.frame.PictureType.I
        else:
            frame.pict_type = av.video.frame.PictureType.NONE

        if self.codec is None:
            try:
                self.codec, self.codec_buffering = create_encoder_context(
                    "hevc_omx", frame.width, frame.height, bitrate=self.target_bitrate
                )
            except Exception:
                self.codec, self.codec_buffering = create_encoder_context(
                    "libx265",
                    frame.width,
                    frame.height,
                    bitrate=self.target_bitrate,
                )

        data_to_send = b""
        for package in self.codec.encode(frame):
            package_bytes = bytes(package)
            if self.codec_buffering:
                if package.pts == self.buffer_pts:
                    self.buffer_data += package_bytes
                else:
                    data_to_send += self.buffer_data
                    self.buffer_data = package_bytes
                    self.buffer_pts = package.pts
            else:
                data_to_send += package_bytes

        if data_to_send:
            yield from self._split_bitstream(data_to_send)

    def encode(
        self, frame: Frame, force_keyframe: bool = False
    ) -> Tuple[List[bytes], int]:
        assert isinstance(frame, av.VideoFrame)
        packages = self._encode_frame(frame, force_keyframe)
        timestamp = convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)
        return self._packetize(packages), timestamp

    def pack(self, packet: Packet) -> Tuple[List[bytes], int]:
        assert isinstance(packet, av.Packet)
        packages = self._split_bitstream(bytes(packet))
        timestamp = convert_timebase(packet.pts, packet.time_base, VIDEO_TIME_BASE)
        return self._packetize(packages), timestamp

    @property
    def target_bitrate(self) -> int:
        return self.__target_bitrate

    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        bitrate = max(MIN_BITRATE, min(bitrate, MAX_BITRATE))
        self.__target_bitrate = bitrate


def h265_depayload(payload: bytes) -> bytes:
    descriptor, data = H265PayloadDescriptor.parse(payload)
    return data


class H265Decoder: ...
