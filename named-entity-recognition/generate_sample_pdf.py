"""Generate a sample PDF with entity-rich content for testing."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
import os


def generate_sample_pdf(output_path: str = "data/sample_document.pdf"):
    """Create the sample PDF used in quick local tests."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='darkblue',
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    title = Paragraph("Business Report: Global Technology Leaders", title_style)
    story.append(title)
    story.append(Spacer(1, 12))
    
    content = [
        {
            "heading": "Technology Industry Overview",
            "text": """
            Apple Inc., headquartered in Cupertino, California, remains one of the most valuable 
            technology companies in the world. Under the leadership of CEO Tim Cook, the company 
            has expanded its product line and services. Microsoft Corporation, based in Redmond, 
            Washington, continues to compete with cloud computing services led by Satya Nadella.
            """
        },
        {
            "heading": "Global Operations",
            "text": """
            Google LLC operates major offices in Mountain View, California, and has expanded 
            significantly into international markets including London, England, and Tokyo, Japan. 
            The company, now part of Alphabet Inc., was founded by Larry Page and Sergey Brin 
            in 1998. Amazon.com Inc., founded by Jeff Bezos in Seattle, Washington, has 
            revolutionized e-commerce and cloud computing through Amazon Web Services.
            """
        },
        {
            "heading": "European Tech Scene",
            "text": """
            Spotify Technology S.A., founded in Stockholm, Sweden by Daniel Ek and Martin Lorentzon, 
            has become the leading music streaming platform. In Germany, SAP SE continues to dominate 
            enterprise software from its headquarters in Walldorf. The European tech ecosystem 
            has seen significant investment from venture capital firms based in Paris, France, 
            and Berlin, Germany.
            """
        },
        {
            "heading": "Financial Performance",
            "text": """
            In the fiscal year 2025, Tesla Inc., led by Elon Musk, reported revenues exceeding 
            $100 billion. The company manufactures electric vehicles at facilities in Fremont, 
            California, and Austin, Texas. Meta Platforms Inc., formerly Facebook, operates 
            under CEO Mark Zuckerberg from headquarters in Menlo Park, California. The company 
            has invested heavily in virtual reality and the metaverse.
            """
        },
        {
            "heading": "Asian Technology Markets",
            "text": """
            Alibaba Group, founded by Jack Ma in Hangzhou, China, has become a dominant force 
            in e-commerce and cloud computing across Asia. Tencent Holdings Limited, based in 
            Shenzhen, China, operates WeChat and is a major player in gaming and social media. 
            In India, companies like Infosys and Tata Consultancy Services, both based in 
            Bangalore, provide IT services to clients worldwide. Samsung Electronics, 
            headquartered in Seoul, South Korea, remains a leading manufacturer of consumer electronics.
            """
        },
        {
            "heading": "Innovation Hubs",
            "text": """
            Silicon Valley, located in the San Francisco Bay Area, continues to be the epicenter 
            of technological innovation. The region is home to Stanford University and the 
            University of California, Berkeley, which produce top engineering talent. Boston, 
            Massachusetts, with institutions like MIT and Harvard University, has emerged as 
            another major tech hub. Austin, Texas, has attracted numerous companies with its 
            favorable business climate and growing talent pool.
            """
        }
    ]
    
    for section in content:
        heading = Paragraph(section["heading"], styles['Heading2'])
        story.append(heading)
        story.append(Spacer(1, 12))
        
        text = Paragraph(section["text"].strip(), styles['Justify'])
        story.append(text)
        story.append(Spacer(1, 20))
    
    footer_text = """
    <i>This document is generated for educational purposes as part of an NLP assignment on 
    Named Entity Recognition. It contains various named entities including organizations, 
    locations, and person names for testing NER models.</i>
    """
    footer = Paragraph(footer_text, styles['Normal'])
    story.append(Spacer(1, 30))
    story.append(footer)
    
    doc.build(story)
    print(f"Sample PDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_sample_pdf()
