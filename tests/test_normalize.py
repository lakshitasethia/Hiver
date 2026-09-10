from support_agent.data.normalize import detect_english, normalize


def test_masks_pii_and_handles():
    m = normalize("Hey @AcmeHelp my order #ABC12345 never came, email me at jo.doe@example.com or +1 415 555 1234")
    assert "<USER>" in m.clean
    assert "<ORDERID>" in m.clean
    assert "<EMAIL>" in m.clean
    assert "<PHONE>" in m.clean
    assert m.masked_spans["<EMAIL>"] == 1
    assert m.raw != m.clean


def test_keeps_raw_intact():
    raw = "  multiple   spaces\n\nand lines  "
    m = normalize(raw)
    assert m.raw == raw
    assert "  " not in m.clean


def test_signature_stripped():
    m = normalize("Please help with my login\n-- Alex\nSent from my iPhone")
    assert "Sent from my" not in m.clean
    assert "login" in m.clean


def test_language_gate():
    assert detect_english("where is my order it has not arrived yet and I am worried")
    assert not detect_english("Bonjour, ma commande n'est jamais arrivee que se passe-t-il vraiment")
    # too short -> assume english, let downstream handle
    assert detect_english("help")


def test_non_english_flag():
    m = normalize("Hola, me cobraron dos veces por el pedido y quiero un reembolso inmediato por favor")
    assert m.is_english is False
    assert m.language == "non-en"
