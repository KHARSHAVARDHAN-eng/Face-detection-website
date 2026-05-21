import React, { useEffect, useState } from 'react';
import { api, STATIC_URL } from '../services/api';
import { Trash2, User } from 'lucide-react';
import toast from 'react-hot-toast';

export default function Dashboard() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchUsers = async () => {
    try {
      const response = await api.get('/users');
      const mapped = response.data.map(u => ({
        ...u,
        image_url: u.image_path ? `${STATIC_URL}/${u.image_path}` : null
      }));
      setUsers(mapped);
    } catch (err) {
      setError('Failed to load users');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const handleDelete = async (id) => {
    if (!window.confirm("Are you sure you want to delete this user?")) return;

    const toastId = toast.loading('Deleting user...');

    try {
      await api.delete(`/user/${id}`);
      setUsers(users.filter((u) => u.id !== id));
      toast.success('User deleted successfully', { id: toastId });
    } catch (err) {
      toast.error('Failed to delete user', { id: toastId });
    }
  };

  if (loading) {
    return (
      <div className="text-center py-20 text-gray-500 font-medium animate-pulse">
        Loading users...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-20 text-red-600">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center bg-white p-6 rounded-xl shadow-sm border border-gray-200">
        <div>
          <h3 className="text-xl font-bold text-gray-900">
            Registered Users
          </h3>

          <p className="text-sm text-gray-500">
            Manage all registered faces in the system.
          </p>
        </div>

        <div className="bg-blue-50 text-blue-700 px-4 py-2 rounded-lg font-bold">
          Total: {users.length}
        </div>
      </div>

      {users.length === 0 ? (
        <div className="text-center py-20 bg-white rounded-xl shadow-sm border border-gray-200 text-gray-500">
          No users registered yet.
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {users.map((user) => (
            <div
              key={user.id}
              className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden flex flex-col hover:shadow-md transition-shadow"
            >
              <div className="aspect-square bg-gray-100 flex items-center justify-center overflow-hidden">
                {user.image_url ? (
                  <img
                    src={user.image_url}
                    alt={user.name}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-gray-250 text-gray-400">
                    <User className="w-16 h-16" />
                  </div>
                )}
              </div>

              <div className="p-4 flex-grow flex flex-col justify-between">
                <div>
                  <h4
                    className="text-lg font-bold text-gray-900 truncate"
                    title={user.name}
                  >
                    {user.name}
                  </h4>

                  <p className="text-xs text-gray-500 mt-1">
                    Added:{' '}
                    {new Date(user.created_at).toLocaleDateString()}
                  </p>

                  <p className="text-xs text-gray-400">
                    ID: #{user.id}
                  </p>
                </div>

                <button
                  onClick={() => handleDelete(user.id)}
                  className="mt-4 w-full flex items-center justify-center px-4 py-2 bg-red-50 text-red-600 hover:bg-red-100 rounded-md transition-colors font-medium text-sm"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}