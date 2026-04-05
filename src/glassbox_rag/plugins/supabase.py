"""
Supabase Database + Vector Store plugin.

Uses supabase-py to provide both relational database and
pgvector-based vector search through a single Supabase project.

Requires: pip install supabase
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from glassbox_rag.plugins.base import DatabasePlugin, VectorStorePlugin
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class SupabaseDatabase(DatabasePlugin):
    """
    Supabase as a relational database plugin.

    Uses the Supabase REST API (PostgREST) for CRUD and
    PostgreSQL full-text search under the hood.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.url = config.get("url", "")
        self.key = config.get("key", "")
        self.schema = config.get("schema", "public")
        self.table = config.get("table", "documents")
        self._client = None

    async def initialize(self) -> bool:
        try:
            connected = await self.connect()
            if connected:
                await self.ensure_tables()
            return connected
        except Exception as e:
            logger.error("Failed to initialize Supabase database: %s", e)
            return False

    async def connect(self) -> bool:
        try:
            from supabase import create_client
            self._client = create_client(self.url, self.key)
            # Verify connection
            self._client.table(self.table).select("id").limit(1).execute()
            logger.info("Supabase database connected: %s", self.url[:40])
            return True
        except ImportError:
            logger.error("supabase not installed. Run: pip install supabase")
            return False
        except Exception as e:
            logger.error("Supabase connection failed: %s", e)
            return False

    async def shutdown(self) -> None:
        self._client = None

    async def health_check(self) -> bool:
        if not self._client:
            return False
        try:
            self._client.table(self.table).select("id").limit(1).execute()
            return True
        except Exception:
            return False

    async def ensure_tables(self) -> None:
        # Supabase tables are managed via dashboard/migrations
        # We just check the table exists
        logger.debug("Supabase tables managed via Supabase dashboard")

    async def insert(self, table: str, data: Dict[str, Any]) -> str:
        if not self._client:
            raise RuntimeError("Supabase not connected")
        record_id = data.get("id", str(uuid4()))
        row = {
            "id": record_id,
            "content": data.get("content", ""),
            "metadata": data.get("metadata", {}),
        }
        self._client.table(table).upsert(row).execute()
        return record_id

    async def update(self, table: str, record_id: str, data: Dict[str, Any]) -> bool:
        if not self._client:
            raise RuntimeError("Supabase not connected")
        resp = self._client.table(table).update(data).eq("id", record_id).execute()
        return len(resp.data) > 0

    async def delete(self, table: str, record_id: str) -> bool:
        if not self._client:
            raise RuntimeError("Supabase not connected")
        resp = self._client.table(table).delete().eq("id", record_id).execute()
        return len(resp.data) > 0

    async def query(
        self, table: str, filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._client:
            raise RuntimeError("Supabase not connected")
        q = self._client.table(table).select("*")
        if filters:
            for k, v in filters.items():
                q = q.eq(k, v)
        resp = q.limit(1000).execute()
        return resp.data

    async def search_text(
        self, terms: List[str], top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Full-text search via Supabase's textSearch filter."""
        if not self._client:
            raise RuntimeError("Supabase not connected")
        query_str = " & ".join(terms)
        resp = (
            self._client.table(self.table)
            .select("*")
            .text_search("content", query_str)
            .limit(top_k)
            .execute()
        )
        return resp.data


class SupabaseVectorStore(VectorStorePlugin):
    """
    Supabase pgvector-based vector store.

    Uses Supabase Edge Functions or RPC calls to the
    `match_documents` PostgreSQL function (standard Supabase
    AI template pattern).

    Requires the following SQL in your Supabase project:

    ```sql
    create extension if not exists vector;

    create table documents (
      id text primary key default gen_random_uuid()::text,
      content text,
      metadata jsonb default '{}',
      embedding vector(1536)
    );

    create or replace function match_documents(
      query_embedding vector(1536),
      match_count int default 5,
      filter jsonb default '{}'
    ) returns table (
      id text,
      content text,
      metadata jsonb,
      similarity float
    )
    language plpgsql as $$
    begin
      return query
      select
        d.id, d.content, d.metadata,
        1 - (d.embedding <=> query_embedding) as similarity
      from documents d
      order by d.embedding <=> query_embedding
      limit match_count;
    end;
    $$;
    ```
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.url = config.get("url", "")
        self.key = config.get("key", "")
        self.table = config.get("table", "documents")
        self.match_fn = config.get("match_function", "match_documents")
        self.vector_dim = config.get("vector_size", 1536)
        self._client = None

    async def initialize(self) -> bool:
        try:
            from supabase import create_client
            self._client = create_client(self.url, self.key)
            logger.info("Supabase vector store connected: %s", self.url[:40])
            return True
        except ImportError:
            logger.error("supabase not installed. Run: pip install supabase")
            return False
        except Exception as e:
            logger.error("Supabase vector store init failed: %s", e)
            return False

    async def shutdown(self) -> None:
        self._client = None

    async def health_check(self) -> bool:
        if not self._client:
            return False
        try:
            self._client.table(self.table).select("id").limit(1).execute()
            return True
        except Exception:
            return False

    async def add_vectors(
        self,
        vectors: List[List[float]],
        ids: Optional[List[str]] = None,
        contents: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        if not self._client:
            raise RuntimeError("Supabase not connected")

        doc_ids: List[str] = []
        rows = []
        for i, vec in enumerate(vectors):
            doc_id = ids[i] if ids else str(uuid4())
            doc_ids.append(doc_id)
            rows.append({
                "id": doc_id,
                "content": contents[i] if contents else "",
                "metadata": metadata[i] if metadata else {},
                "embedding": vec,
            })

        # Batch upsert
        batch_size = 500
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            self._client.table(self.table).upsert(batch).execute()

        return doc_ids

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        if not self._client:
            raise RuntimeError("Supabase not connected")

        resp = self._client.rpc(
            self.match_fn,
            {
                "query_embedding": query_vector,
                "match_count": top_k,
                "filter": kwargs.get("filter", {}),
            },
        ).execute()

        results = []
        for row in resp.data:
            results.append((
                row["id"],
                row.get("similarity", 0.0),
                row.get("content", ""),
                row.get("metadata", {}),
            ))
        return results

    async def get_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        if not self._client:
            return None
        resp = (
            self._client.table(self.table)
            .select("id, content, metadata")
            .eq("id", vector_id)
            .limit(1)
            .execute()
        )
        return resp.data[0] if resp.data else None

    async def delete_vector(self, vector_id: str) -> bool:
        if not self._client:
            return False
        resp = self._client.table(self.table).delete().eq("id", vector_id).execute()
        return len(resp.data) > 0

    async def count(self) -> int:
        if not self._client:
            return 0
        resp = self._client.table(self.table).select("id", count="exact").execute()
        return resp.count or 0
