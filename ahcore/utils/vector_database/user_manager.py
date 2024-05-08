from functools import wraps
from logging import getLogger
from typing import Any, Callable, TypeVar, cast

from pymilvus import Collection, MilvusException, Role, connections, utility

logger = getLogger(__name__)


T = TypeVar("T", bound=Callable[..., Any])


def handle_milvus_error(func: T) -> T:
    """Decorator to handle exceptions raised by Milvus operations."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except MilvusException as e:
            logger.error(f"Milvus error in {func.__name__}: {e}")
            return None

    return cast(T, wrapper)


class MilvusUserManager:
    """Lightweight wrapper around pymilvus to manage users, roles, and privileges."""

    def __init__(self, host: str, port: int, user: str, password: str, database: str = "default"):
        """
        Initializes MilvusUserManager with host, port, and user credentials.
        Should be an admin user to manage roles and users effectively.
        """
        self.host = host
        self.port = port
        self.user = user
        self._password = password
        self.database = database
        self._collection = None
        connections.connect(host=self.host, port=self.port, user=self.user, password=self._password)
        logger.info("Connected to Milvus successfully.")

    @property
    def collection(self) -> Collection:
        """Gets the current working collection."""
        return self._collection

    @collection.setter
    def collection(self, value: str) -> None:
        """Sets the current working collection by name."""
        try:
            new_collection = Collection(name=value)
            self._collection = new_collection
            logger.info(f"Set collection to {value}")
        except Exception as e:
            logger.error(f"Error setting collection: {e}")

    @handle_milvus_error
    def create_user(self, username: str, password: str) -> None:
        """Creates a new user in Milvus."""
        utility.create_user(username, password, using=self.database)
        logger.info(f"User {username} created successfully.")

    @handle_milvus_error
    def change_password(self, username: str, old_password: str, new_password: str) -> None:
        """Changes the password of a user."""
        utility.reset_password(username, old_password, new_password, using=self.database)
        logger.info(f"Password for user {username} changed successfully.")

    @handle_milvus_error
    def assign_user_to_role(self, username: str, role_name: str) -> None:
        """Assigns a user to a role for the current collection."""
        if self.collection:
            role = Role(role_name, using=self.database)
            role.add_user(username)
            logger.info(f"Assigned {role_name} to {username} on collection {self.collection.name}")
        else:
            logger.info("No collection set. Please set a collection before assigning roles.")

    @handle_milvus_error
    def remove_user_from_role(self, username: str, role_name: str) -> None:
        """Removes a user from a role for the current collection."""
        if self.collection:
            role = Role(role_name, using=self.database)
            role.remove_user(username)
            logger.info(f"Removed {role_name} from {username} on collection {self.collection.name}")
        else:
            logger.info("No collection set. Please set a collection before removing roles.")

    @handle_milvus_error
    def create_role(self, role_name: str) -> None:
        """Creates a new role."""
        role = Role(role_name, using=self.database)
        role.create()
        logger.info(f"Role {role_name} created successfully.")

    @handle_milvus_error
    def drop_role(self, role_name: str) -> None:
        """Drops a role."""
        role = Role(role_name, using=self.database)
        role.drop()
        logger.info(f"Role {role_name} dropped.")

    @handle_milvus_error
    def role_info(self, role_name: str) -> None:
        """Gets information about a role."""
        role = Role(role_name, using=self.database)
        logger.info(f"Role {role_name} has the following users: {role.get_users()}")
        logger.info(f"Role {role_name} has the following grants: {role.list_grants()}")
        current_grants = role.list_grant("Collection", self.collection._name)
        logger.info(f"Role {role} has the following grants on collection {self.collection._name}: {current_grants}")

    @handle_milvus_error
    def user_info(self, username: str) -> None:
        """Gets information about a user."""
        user_info = utility.list_user(username, include_role_info=True, using=self.database)
        logger.info(f"User info for {username}: {user_info}")

    def __del__(self) -> None:
        """Ensures that the Milvus connection is closed when the object is deleted."""
        connections.disconnect()
        logger.info("Disconnected from Milvus.")
