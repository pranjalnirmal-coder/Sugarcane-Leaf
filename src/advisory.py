"""Preliminary, source-linked sugarcane disease guidance for supported labels."""
from __future__ import annotations


ADVISORIES = {
    "Healthy": {
        "display_name": "Healthy",
        "description": "The image resembles the dataset's healthy-leaf examples. This does not certify that the whole plant or field is disease-free.",
        "prevention": ["Use healthy planting material and keep tools and field operations clean.",
                       "Inspect plants regularly and seek expert advice if symptoms appear."],
        "management": ["No disease-specific treatment is indicated by this image classification."],
        "expert": "Consult a local agricultural expert if symptoms or crop decline are present despite this result.",
        "sources": ["https://agritech.tnau.ac.in/crop_protection/crop_prot_ipm_sugarcane.html"],
    },
    "Mosaic": {
        "display_name": "Mosaic",
        "description": "The dataset label is Mosaic. Sugarcane mosaic can cause mottled or irregular light and dark green areas, but visual similarity alone is not a definitive diagnosis.",
        "prevention": ["Use healthy, disease-free planting material.",
                       "Monitor the crop and ask an agricultural expert to confirm suspected virus symptoms."],
        "management": ["Have suspected plants assessed by a sugarcane disease specialist before taking field-level action."],
        "expert": "Consult an agricultural expert for confirmation and locally appropriate management, especially if symptoms are spreading.",
        "sources": ["https://agritech.tnau.ac.in/org_farm/orgfarm_agridiseases.html"],
    },
    "RedRot": {
        "display_name": "Red Rot",
        "description": "The dataset label is Red Rot. Reported signs can include leaf yellowing and drying; confirmation often requires inspecting cane tissue, so a leaf photo alone is limited.",
        "prevention": ["Use setts from healthy, disease-free planting stock.",
                       "Use locally recommended resistant varieties and maintain field sanitation."],
        "management": ["Remove and manage suspected affected clumps only under local expert guidance.",
                        "Avoid reusing planting material from suspected diseased canes."],
        "expert": "Promptly contact a sugarcane extension officer or plant pathologist to confirm red rot and plan field management.",
        "sources": ["https://www.agritech.tnau.ac.in/govt_schemes_services/aas/sugarcane.html",
                    "https://agritech.tnau.ac.in/expert_system/sugar/faq.html"],
    },
    "Rust": {
        "display_name": "Rust/Brown Rust",
        "description": "The report mentions Brown Rust. The dataset folder is named Rust; typical sugarcane rust can produce elongated yellow to reddish-brown leaf spots and a rusty appearance.",
        "prevention": ["Use varieties recommended locally for resistance and inspect leaves for developing spots.",
                       "Keep records of affected areas and seek local confirmation."],
        "management": ["Ask an agricultural expert to confirm the disease and determine whether any locally registered treatment is appropriate."],
        "expert": "Consult an agricultural extension officer when rust-like symptoms spread or affect multiple plants.",
        "sources": ["https://www.agritech.tnau.ac.in/crop_protection/sugarcane_diseases/sugarcane_d3.html"],
    },
    "Yellow": {
        "display_name": "Yellow",
        "description": "Yellow is the dataset folder name; it does not identify a specific disease. Yellowing can have several causes, and this app cannot equate this label with Sugarcane Yellow Leaf Disease.",
        "prevention": ["Photograph the whole plant and surrounding crop, and note when symptoms began.",
                       "Use clean planting material and maintain regular crop monitoring."],
        "management": ["Do not apply a pesticide based only on this image label. Diagnosis may require field or laboratory assessment."],
        "expert": "Have an agricultural expert assess yellowing, especially if it is spreading, includes midrib changes, or accompanies stunting.",
        "sources": ["https://agritech.tnau.ac.in/crop_protection/sugarcane_diseases/sugarcane_d1.html"],
    },
}


def get_advisory(class_name: str) -> dict:
    """Return advisory data only for the actual saved model class label."""
    try:
        return ADVISORIES[class_name]
    except KeyError as exc:
        raise KeyError(f"No advisory is configured for model class {class_name!r}.") from exc


def validate_advisories(class_names: list[str]) -> None:
    missing = sorted(set(class_names) - set(ADVISORIES))
    if missing:
        raise ValueError(f"Advisories are missing for model classes: {missing}")
