"""Virtual gantry state and keyboard commands, matching holosoma run_sim."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np
import numpy.typing as npt

LOG = logging.getLogger(__name__)


class GantryCommand(Enum):
    LENGTH_ADJUST = "gantry_length_adjust"
    TOGGLE = "gantry_toggle"
    FORCE_ADJUST = "gantry_force_adjust"
    FORCE_SIGN_TOGGLE = "gantry_force_sign_toggle"


@dataclass(frozen=True)
class GantryCommandData:
    command: GantryCommand
    parameters: dict[str, float | bool] = field(default_factory=dict)


# GLFW keycodes used by MuJoCo's passive viewer. Keeping these values here lets
# headless tests exercise the exact same command path without opening a window.
KEY_0 = 48
KEY_7 = 55
KEY_8 = 56
KEY_9 = 57
KEY_MINUS = 45

GANTRY_KEY_COMMANDS = {
    KEY_7: GantryCommandData(GantryCommand.LENGTH_ADJUST, {"amount": -0.1}),
    KEY_8: GantryCommandData(GantryCommand.LENGTH_ADJUST, {"amount": 0.1}),
    KEY_9: GantryCommandData(GantryCommand.TOGGLE),
    KEY_0: GantryCommandData(GantryCommand.FORCE_ADJUST),
    KEY_MINUS: GantryCommandData(GantryCommand.FORCE_SIGN_TOGGLE),
}


@dataclass
class VirtualGantry:
    enabled: bool = True
    stiffness: float = 200.0
    damping: float = 100.0
    height: float = 3.0
    length: float = 0.0
    apply_force: float = 0.0
    apply_force_sign: int = -1
    point: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 0.0, 3.0]))

    def set_position_to_robot(self, robot_position: npt.ArrayLike) -> None:
        position = np.asarray(robot_position, dtype=np.float64)
        self.point = np.array([position[0], position[1], self.height], dtype=np.float64)

    def set_enable(self, enabled: bool | None = None) -> None:
        self.enabled = not self.enabled if enabled is None else enabled

    def handle_command(self, command_data: GantryCommandData | GantryCommand) -> bool:
        if isinstance(command_data, GantryCommand):
            command_data = GantryCommandData(command_data)
        command = command_data.command
        parameters = command_data.parameters

        if command == GantryCommand.LENGTH_ADJUST:
            amount = float(parameters.get("amount", 0.1))
            self.length += amount
            LOG.info("Gantry length adjusted by %s to %.2f", amount, self.length)
            return True
        if command == GantryCommand.TOGGLE:
            requested = parameters.get("enabled")
            self.set_enable(None if requested is None else bool(requested))
            LOG.info("Gantry %s", "enabled" if self.enabled else "disabled")
            return True
        if command == GantryCommand.FORCE_ADJUST:
            amount = float(parameters.get("amount", 10 * self.apply_force_sign))
            self.apply_force = float(np.clip(self.apply_force + amount, -100.0, 100.0))
            LOG.info("Gantry apply_force adjusted by %s to %s", amount, self.apply_force)
            return True
        if command == GantryCommand.FORCE_SIGN_TOGGLE:
            self.apply_force_sign *= -1
            LOG.info("Gantry force sign toggled to %d", self.apply_force_sign)
            return True
        return False

    def handle_key(self, keycode: int) -> bool:
        command = GANTRY_KEY_COMMANDS.get(keycode)
        return command is not None and self.handle_command(command)

    def advance(
        self,
        position: npt.NDArray[np.float64],
        velocity: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return the spring-damper force using holosoma's gantry equation."""
        delta = self.point - position
        distance = np.linalg.norm(delta)
        if distance < 1e-9:
            return np.zeros(3, dtype=np.float64)
        direction = delta / distance
        radial_velocity = float(np.dot(velocity, direction))
        magnitude = self.stiffness * (distance - self.length) - self.damping * radial_velocity
        return magnitude * direction

