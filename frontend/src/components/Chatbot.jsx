import { useState } from "react";
import api from "../services/api";

export default function Chatbot() {
  const [q, setQ] = useState("");
  const [a, setA] = useState("");

  const ask = async () => {
    const res = await api.post("/chat/ask?question=" + q);
    setA(res.data.answer);
  };

  return (
    <div className="chatbot">
      <input placeholder="Ask me anything…" onChange={e=>setQ(e.target.value)} />
      <button onClick={ask}>Send</button>
      <p>{a}</p>
    </div>
  );
}
