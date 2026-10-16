from typing import Optional
from fastapi import Header, HTTPException, status, Depends
from backend.app.core.config import settings

def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """
    Extracts user details from Authorization header if provided.
    Does not reject unauthenticated requests (allows guest users).
    """
    if not authorization:
        return None
    try:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
            # If Supabase is configured with a secret or public key, we can decode JWT
            # For lightweight inspection without requiring external calls:
            import jwt
            decoded = jwt.decode(token, options={"verify_signature": False})
            return {
                "id": decoded.get("sub", decoded.get("id")),
                "email": decoded.get("email"),
                "role": decoded.get("role", decoded.get("user_metadata", {}).get("role", "authenticated"))
            }
    except Exception:
        pass
    return None

def verify_admin_access(
    authorization: Optional[str] = Header(None),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")
) -> dict:
    """
    Enforces server-side authorization for admin-only endpoints.
    Checks either X-Admin-Key or Supabase Bearer token with admin role.
    """
    # 1. Check direct admin secret header
    if x_admin_key and x_admin_key == settings.ADMIN_SECRET_KEY:
        return {"id": "admin-system", "role": "admin", "type": "api_key"}

    # 2. Check Bearer token matching ADMIN_SECRET_KEY
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
            if token == settings.ADMIN_SECRET_KEY:
                return {"id": "admin-system", "role": "admin", "type": "bearer_key"}
            
            # Check Supabase JWT role claim
            try:
                import jwt
                decoded = jwt.decode(token, options={"verify_signature": False})
                user_role = decoded.get("role") or decoded.get("user_metadata", {}).get("role") or decoded.get("app_metadata", {}).get("role")
                if user_role == "admin":
                    return {
                        "id": decoded.get("sub", decoded.get("id")),
                        "email": decoded.get("email"),
                        "role": "admin",
                        "type": "supabase_jwt"
                    }
            except Exception:
                pass

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Administrator privileges required to access this resource."
    )
