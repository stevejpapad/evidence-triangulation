### Claim Reformulation    
def prompt_1(user_post, image_urls):

    prompt = f"""You are a content analysis assistant. Your task is to infer the claim being made in a social media post in relation to an image. 
    Extract the clearest version of the claim. Rewrite it as a short sentence that captures the main idea.
    Return only the extracted claim, no additional explanation. If no clear claim is being made, return 'None'. 
    
    ### INPUT
    User Post: {user_post}
    Image: """

    user_content = {
        "user_prompt":prompt,
        "image_urls": image_urls
    }

    return user_content
    



### Multimodal Context Extraction
def prompt_2(user_claim, image_urls):
    
    prompt = f"""You are a fact-checking assistant analyzing a social media post that includes a text and an image. 
    Your task is to extract structured information from the post.
    Please extract the following features and return them in JSON format:
    1. image_description: A concise description of what is visible in the image.
    2. ocr_text: Any readable text found inside the image, such as signs, headlines, or memes.
    3. named_entities: A list of named entities found in the post text and image, such as people, places, organizations, dates.
    4. five_Ws: Summarize what can be inferred about who, what, when, where, why -- based only on the post text and image.
    Only extract what is clearly present. If a feature is not available, return 'None' for that field.
    Respond in this exact format:
    {{
      "named_entities": [...],    
      "image_description": "...",
      "ocr_text": "...",
      "five_Ws": "..."
    }}

    ### INPUT
    Claim: {user_claim}
    Image: """

    user_content = {
        "user_prompt": prompt,
        "image_urls": image_urls
    }


    return user_content 


# Search Query Generation
def prompt_3(claim, image_urls, features):

    context_lines = []
    if features:
        if desc := features.get("image_description", "").strip():
            context_lines.append(f"-Image Description: {desc}")
        if ocr := features.get("ocr_text", "").strip():
            context_lines.append(f"-OCR Text: {ocr}")
        if ents := features.get("named_entities"):
            if isinstance(ents, list) and ents:
                context_lines.append(f"- Named Entities: {', '.join(ents)}")
        if fivew := features.get("five_Ws", "").strip():
            context_lines.append(f"- Five Ws: {fivew}")

    multimodal_context = "\n    ".join(context_lines) if context_lines else "No multimodal context is available."

    prompt = f"""You are a fact-checking assistant.
    Given a social media post and an extracted claim, your task is to generate 3 targeted search queries and 1 image search query to help verify the claim. Use the claim as the basis for all queries. The goal is to help a human fact-checker find reliable evidence (e.g. news articles, press releases, official records) that can confirm or refute the claim.
    You will receive:
    - The claim extracted from the post's text
    - The attached image
    - Extracted multimodal context: named entities, image description, OCR text, and the 'Five Ws' (Who, What, When, Where, Why)
    Guidelines:
    - Base the queries on all available information: post claim, images, and extracted multimodal context.
    - Use exact phrases in quotes when relevant.
    - Focus on names, dates, locations, numeric values, or other verifiable details.
    - Avoid vague or generic queries, focus on language that would return reliable sources like news articles, public statements, or official documents.
    - Use visual or OCR content when relevant (e.g., embedded text, people, logos, official documents).
    - The goal is to help a fact-checker quickly find reliable, relevant information to verify the post.  
    Return the output in the following JSON format:
    {{
    "text_queries": [
        "query 1", 
        "query 2",
        "query 3"
    ],
    "image_search_query": "image-related query"
    }}

    ### INPUTS
    User claim: {claim}
    Multimodal Context: {multimodal_context}
    Image: """

    user_content = {
        "user_prompt": prompt,
        "image_urls": image_urls
    }


    return user_content


# Evidence Excerpt Extraction
def prompt_4(user_post, article):
    return f"""Given a user post (image and text) and an external article, extract only the few most relevant sentences or passages from the article that help verify or refute the user post.
    - If the article does not contain any relevant information, return exactly: None
    - Do not summarize. Copy the sentences directly from the article.
    - Keep only the most relevant 1, 2, or 3 sentences maximum, not the whole article.

    Claim:"{user_post}"
    External article:{article}    
    Return the most relevant sentences (or "None" if no relevant evidence is found): """

# Instructions for QWEN GTE feature extraction
def prompt_5():
    return "Given a post and an article, highlight the aspects of the article that are most relevant to the post."


# Detection prompt for VLMs
def prompt_6(user_post, article_excerpts_t2t, article_excerpts_i2t):
    ### The image is provided directly in the OpenRouter api
    
    evidence_block = ""
    if article_excerpts_t2t or article_excerpts_i2t:
        evidence_block = "\nExternal Evidence:"
        if article_excerpts_t2t:
            evidence_block += f"\n- Text-based Search Results: {article_excerpts_t2t}"
        if article_excerpts_i2t:
            evidence_block += f"\n- Image-based Search Results: {article_excerpts_i2t}"

    return f"""You are given a social media post consisting of text and an image.

    Classify the post into exactly one of the following categories:
    - accurate: The text and image are consistent, and the post is in no way misleading.
    - misleading: The post presents a false or incorrect impression, including cases where the image is reused from another time or place, misidentifies people or events, or provides wrong context.

    Rules:
    - If External Evidence is available, compare the Text and Image against them.
    - Respond with **only one word**: "accurate" or "misleading".
    - Do NOT provide explanations, reasoning, or additional text.

    Post Text: {user_post} {evidence_block} """

