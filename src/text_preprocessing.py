# Text preprocessing - Customer Review Intent Classification

import re
import string
from unicodedata import normalize

import contractions


def countWords(my_text):
    """Count the number of words in a string."""
    res = len(re.findall(r"\w+", my_text))
    return res


def textNormalize(my_text: string) -> string:
    """Lowercase and strip accents/diacritics."""
    my_text = my_text.lower()
    return normalize("NFKD", my_text).encode("ascii", errors="ignore").decode("utf-8")


def inputDataContractions(my_text, my_slang=False):
    """Expand contractions (e.g. "don't" -> "do not")."""
    return contractions.fix(my_text, slang=my_slang)


def regularExpressionTextCleaning(my_text: string) -> string:
    """Clean boilerplate, disclaimers, URLs, emails, emojis, and symbols."""
    text = my_text

    # Strip common email greetings
    text = text.replace("follow us on ", "")
    text = text.replace("good morning", "")
    text = text.replace("good afternoon", "")
    text = text.replace("good evening", "")
    text = text.replace("good day", "")

    # Strip email forwarding/reply prefixes
    text = text.replace("re:", "")
    text = text.replace("fwd:", "")
    text = text.replace("ext:", "")
    text = text.replace("attn", "")
    text = text.replace("disclaimer", "")
    text = text.replace("***** please do not respond to this email *****", "")
    text = text.replace("-----original message-----", "")

    # Strip common email disclaimers and confidentiality notices
    _disclaimers = [
        "alert: this email originated from outside. do not click links or open attachments unless you know the sender and trust the content is safe.",
        "this message contains confidential information intended only for the person named above. if you have received this message in error, please notify the sender immediately by replying to this e-mail. if you are not the intended recipient you must not use, disclose, distribute, copy, or print this e-mail. thank you.",
        "the information contained in this communication from the sender is confidential. it is intended solely for use by the recipient and others authorized to receive it. if you are not the intended recipient, you are hereby notified that any disclosure, copying, use, or distribution of the information included in this email is prohibited and may be unlawful.",
        "confidentiality notice: the information contained with this transmission are the private, confidential property of the sender, and the material is privileged communication intended solely for the individual(s) indicated. if you are not the intended recipient, you are hereby notified that any review, disclosure, copying, distribution or the taking of any other action relevant to the contents of this transmission are strictly prohibited. If you have received this transmission in error, please contact the sender by reply email and destroy all copies of the original message.",
        "the information in this e-mail is confidential. it is intended for the exclusive use of the individual or entity to whom it is addressed. this message may contain information that is confidential. if the reader of this message is not the intended recipient, be aware that any disclosure, dissemination, distribution or copying of this communication, or the use of its contents, is prohibited. if you have received this email in error, please notify the sender and delete this email.",
        "this message and any attached documents are only for the use of the intended recipient(s), are confidential and may contain privileged information. any unauthorized review, use, retransmission, or other disclosure is strictly prohibited. if you have received this message in error, notify the sender immediately, and delete the original message.",
        "all rights reserved",
        "the entered text is",
        "please do not reply directly to this email. this email address is not monitored.",
        "if you have received this message in error, or if there is a problem with the communication, please notify the sender immediately and destroy all copies of this e-mail and any attachments. the unauthorized use, disclosure, reproduction, forwarding, copying or alteration of this message is strictly prohibited and may be unlawful.",
        "if you wish to no longer receive electronic messages from this sender, please respond and advise accordingly in your return email.",
        "this message was system generated.",
        "if you have any questions, you can contact our team at",
        "if this is your first time logging into the payer direct hub you should have received a temporary password via email so you can complete the free registration process. ",
        "note: some browser security settings may prevent you from accessing the URL directly if you click on it so you may need to copy the URL text and paste it into your browsers col2_name field.",
        " please do not reply to this automated message ",
        "let us know if you have questions or concerns",
        "please consider the environment before printing this email.",
        "e-mail messages may contain viruses, worms, or other malicious code. by reading the message and opening any attachments, the recipient accepts full responsibility for taking protective action against such code. Henry Schein is not liable for any loss or damage arising from this message.",
        "the information in this email is confidential and may be legally privileged. it is intended solely for the col2_nameee(s). access to this e-mail by anyone else is unauthorized.",
        "[external]",
        "report suspicious",
        "caution: this email originated outside of the company. do not click on links or open attachments unless you have authenticated the sender.",
        "for more information about the handling of your personal data, please click on the following link:",
        "if you do not have a password or need assistance, our support team is here to help.",
        "note: if you no longer wish to receive this notification, please contact your administrator.",
        "--confidentiality notice: this e-mail message, including any attachments, is for the sole use of the intended recipient(s) and may contain confidential, proprietary, and/or privileged information protected by law. if you are not the intended recipient, you may not read, use, copy, or distribute this e-mail message or its attachments. if you believe you have received this e-mail message in error, please contact the sender by reply e-mail or telephone immediately and destroy all copies of the original message.",
        "confidentiality notice - this e-mail transmission, and any documents, files or previous e-mail messages attached to it, may contain information that is confidential and/or proprietary or legally privileged. If you are not the intended recipient, or a person responsible for delivering it to the intended recipient, you are hereby notified that you must not read or play this transmission and that any disclosure, copying, printing, distribution or use of any of the information contained in or attached to this transmission is strictly prohibited. if you have received this transmission in error, please immediately notify the sender by telephone or return e-mail and delete the original transmission and its attachments without reading or saving in any manner. thank you.",
        "if you need adobe reader to view the pdf document, you can download the latest version from http://adobe.com free of charge.",
        "this message is from an external sender. be cautious, especially with links and attachments. ",
        "this message is intended for the exclusive use of the intended addressed. if you have received this message in error or are not the intended col2_nameee or his or her authorized agent, please notify me immediately by e-mail, discard any paper copies and delete all electronic files of this message.",
        "if you have concerns about the validity of this message, please contact the sender directly",
        "if you require assistance opening this message, please click here.",
    ]
    for disclaimer in _disclaimers:
        text = text.replace(disclaimer, "")

    # Common abbreviations
    text = text.replace("deposited", "transferred")
    text = text.replace("add wtr", "added water")
    text = text.replace("cl ", "credit limit ")

    if not text:
        return ""

    # Remove URLs
    _url_re = re.compile(
        r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)"
        r"(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+"
        r"(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?]))"
    )
    text = _url_re.sub("", text)

    # Remove email addresses
    _email_re = re.compile(r"[a-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+")
    text = _email_re.sub("", text)

    # Remove emojis
    _emoji_re = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = _emoji_re.sub("", text)

    # Normalize punctuation and special characters
    text = re.sub(r"\:\)|\:\(", " ", text)
    text = re.sub(r"([a-zA-Z]+)\?s ", r"\1 's ", text)
    text = re.sub(r"('s|'d)", r" \1", text)
    text = re.sub(r'([$><+@{}!?()/;,%\^\[\]-])', r" \1 ", text)
    text = re.sub(r"([\-:$><+@{}!?()/;,%\^])", r" \1 ", text)
    text = re.sub(r"([^&])([&])([^&])", r"\1 \2 \3", text)
    text = re.sub(
        r"&quot;|&lt;|&gt;|&lsquo;|&rsquo;|&ldquo;|&rdquo;|&nbsp;|&amp;|&apos;|&cent;|&pound;|&yen;|&euro;|&copy;|&reg;",
        " ",
        text,
    )
    text = re.sub(r"[\(|\{|\[]\s*?[\)|\}|\]]", " ", text)
    text = re.sub(r"'(?!(s|d))|^'", " ", text)
    text = re.sub("\n", " ", text)
    text = re.sub(r":", " ", text)

    # Remove emoticon patterns
    _happy_re = re.compile(
        r" \uFF3C\(\^o\^\)\uFF0F|\:\-\)|\:\)|\;\)|\:o\)| \:\]| \:3| \:c\)| \:\>|\=\]|8\)| \=\)| \:}|\:\^\)| \:\-D| \:D|8\-D|8D|x\-D|xD|X\-D|XD| \=\-D| \=D|\=\-3| \=3| \:\-\)\)| \:\'\-\)| \:\'\)| \:\*| \:\^\*| \>\:P| \:\-P| \:P|X\-P|x\-p| xp| XP|\:\-p|\:p|\=p|\:\-b|\:b| \>\:\)| \>\;\)| \>\:\-\)|\<3 "
    )
    text = _happy_re.sub("", text)

    _sad_re = re.compile(
        r"\=/\/\|;\(|>\:?\\*|\:\{|\:c|\:\-c|\:'\-\(|>.*?<|:\(|>\:\(|=\/|\:L|\:-/|\>:/|\:S|\:\[|\:\-\|\|\:\-\)|\:\-\|\||\=L|\:<|\:\-\[|\:\-<|=/\/\|=\/|>\:\(|\:\(|\:'\-\(|\:'\(|\:?\\*|\=?\\?"
    )
    text = _sad_re.sub("", text)

    # Remove remaining symbols (with comma removal)
    _symbols_re = re.compile(r"\{|\}|\:|\\|/|\[|\]|\+|\<|\>|\_\u2022|\u00AE|\*|\u201C|\u201D|\"|\!|\^|\u2191|\u00AE|\u274F|\u2192|\$|\--|\||,")
    text = _symbols_re.sub("", text)

    # Keep only alphanumeric, spaces, newlines, and periods
    text = re.sub("[^a-zA-Z0-9 \n.]", " ", text)

    # Remove miscellaneous symbols
    text = text.translate({ord(c): None for c in "~--\t._!\u00A7*\t\t"})

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()
