from __future__ import annotations

from typing import Callable, Dict, List
import logging

from .types_ import Channel, ChannelConfig

logger = logging.getLogger(__name__)


class ChannelManager:
    """
    ChannelManager: build and manage concrete Channel instances from ChannelConfig configs.

    用法示例（结合 message.config_runtime.setup_from_config）:

        from message.config_runtime import setup_from_config
        from channels.channel_manager import ChannelManager

        cfg = setup_from_config()
        if cfg:
            mgr, bindings, accounts = cfg
            ch_mgr = ChannelManager()
            ch_mgr.register_builtin_channels()
            channels = ch_mgr.build_from_accounts(accounts)
            # 把 channels 交给 ChannelBridge 轮询
    """

    def __init__(self) -> None:
        # channel_type -> factory(ChannelConfig) -> Channel
        self._factories: Dict[str, Callable[[ChannelConfig], Channel]] = {}
        self._channels: List[Channel] = []

    # ------------------------------------------------------------------
    # Factory registration
    # ------------------------------------------------------------------
    def register_factory(
        self,
        channel_type: str,
        factory: Callable[[ChannelConfig], Channel],
    ) -> None:
        """Register a factory for a given channel type string (e.g. 'whatsapp_web')."""
        self._factories[channel_type] = factory

    def register_builtin_channels(self, types: List[str] | None = None) -> None:
        """
        Register built-in channels using simple string checks (no dynamic class-name magic).

        types: 可选，仅当配置中实际出现了这些类型时才去导入对应的实现，
               避免无意义的 import 尝试。比如 ["whatsapp", "whatsapp_web"]。
               如果为 None，则尝试注册当前支持的所有内置类型。
        """
        wanted = set(t.strip() for t in (types or []) if t.strip()) or {
            "whatsapp",
            "whatsapp_web",
            "cli",
        }

        # CLI channel: purely local stdin/stdout, does not need ChannelConfig
        if "cli" in wanted:
            try:
                from .types_ import CLIChannel
            except Exception as exc:  # pragma: no cover
                logger.debug("CLIChannel not available: %s", exc)
            else:
                # ignore ChannelConfig, CLIChannel is process-local
                self.register_factory("cli", lambda _acc: CLIChannel())

        if "whatsapp" in wanted:
            try:
                from .whatsapp import WhatsAppChannel
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.debug("WhatsAppChannel not available: %s", exc)
            else:
                self.register_factory("whatsapp", WhatsAppChannel)

        if "whatsapp_web" in wanted:
            try:
                from .whatsapp_web import WhatsAppWebChannel
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.debug("WhatsAppWebChannel not available: %s", exc)
            else:
                self.register_factory("whatsapp_web", WhatsAppWebChannel)

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------
    @property
    def channels(self) -> List[Channel]:
        """Return the list of instantiated channels."""
        return list(self._channels)

    def add_channel_from_config(self, account: ChannelConfig) -> Channel | None:
        """
        Create a channel from a single ChannelConfig and add it to the managed list.
        Registers the builtin factory for the account's channel type if not already registered.
        Returns the new channel, or None if the type is unsupported or construction fails.
        """
        ch_type = (account.channel or "").strip()
        if not ch_type:
            logger.warning("add_channel_from_config: account has no channel type")
            return None
        if ch_type not in self._factories:
            self.register_builtin_channels([ch_type])
        factory = self._factories.get(ch_type)
        if not factory:
            logger.info(
                "No channel factory for type '%s'; skipping account_id=%s",
                ch_type,
                getattr(account, "account_id", ""),
            )
            return None
        try:
            ch = factory(account)
        except Exception as exc:
            logger.error(
                "Failed to construct channel '%s' (%s): %s",
                ch_type,
                getattr(account, "account_id", ""),
                exc,
            )
            return None
        self._channels.append(ch)
        logger.info("Channel added: %s (%s)", ch_type, getattr(account, "account_id", ""))
        return ch

    def build_from_accounts(self, accounts: List[ChannelConfig]) -> List[Channel]:
        """
        Given a list of ChannelConfig configs, register needed factories
        and construct Channel instances.

        步骤：
        1. 从 accounts 中统计出现过的 channel 类型（如 whatsapp_web）。
        2. 调用 register_builtin_channels(types) 用 if/else 注册已支持的内置渠道。
        3. 对每个 account，如果有对应 factory，则实例化并加入列表。
        """
        # 1) 根据配置中实际出现的类型，按字符串判断注册内置渠道
        types = {acc.channel for acc in accounts if getattr(acc, "channel", "").strip()}
        self.register_builtin_channels(list(types))

        # 2) 实例化渠道
        self._channels = []
        for acc in accounts:
            ch_type = (acc.channel or "").strip()
            factory = self._factories.get(ch_type)
            if not factory:
                logger.info("No channel factory registered for type '%s'; skipping account_id=%s", ch_type, getattr(acc, "account_id", ""))
                continue
            try:
                ch = factory(acc)
            except Exception as exc:
                logger.error("Failed to construct channel '%s' (%s): %s", ch_type, getattr(acc, "account_id", ""), exc)
                continue
            self._channels.append(ch)
            logger.info("Channel instantiated: %s (%s)", ch_type, getattr(acc, "account_id", ""))
        return list(self._channels)

    def close_all(self) -> None:
        """Close all instantiated channels."""
        for ch in self._channels:
            try:
                ch.close()
            except Exception:
                pass
        self._channels.clear()

