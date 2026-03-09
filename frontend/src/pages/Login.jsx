import api from "../services/api";

export default function Login() {
  const login = async () => {
    await api.post("/auth/login?username=admin&password=admin");
    window.location.href = "/upload";
  };

  return (
    <div className="center-card">
  <h2>🌾 Agricultural Advisory</h2>
  <p>Sign in to continue</p>
  <button onClick={login}>Login</button>
</div>
  );
}
