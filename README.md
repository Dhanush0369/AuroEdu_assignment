# **AuroEdu Assignment**  

## **Process Overview**  

1. **Connect to the PostgreSQL database** and define the required table schema.  
2. **Implement the `add_document` API** to add documents.  
3. **Generate vector embeddings** for the documents and store them in the database.  
4. **Implement the `select_documents` API** to allow users to specify document preferences.  
5. **Implement the `ask_question` API** to handle user queries efficiently.  

## **Status**  

✅ All APIs have been tested and are running perfectly.  

## **Future Improvement**  

There is a more efficient approach where **PGVector** can automate the embedding process, eliminating the need to store embeddings manually. However, due to time constraints, this has not been implemented yet.  
