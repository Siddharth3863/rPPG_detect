import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
    return (
        <div className="container mt-5">
            <h1 className="text-center">Home Page</h1>
            <div className="card mx-auto" style={{ maxWidth: '300px' }}>
                <div className="card-body">
                    <h2 className="card-title text-center">Options:</h2>
                    <ul className="list-group list-group-flush">
                        <li className="list-group-item">
                            <Link to="/record-session-pfe" className="btn btn-primary btn-block">Using PFE/TFA</Link>
                        </li>
                        <li className="list-group-item">
                            <Link to="/record-session-phys" className="btn btn-primary btn-block">Using PhysNet</Link>
                        </li>
                        <li className="list-group-item">
                            <Link to="/record-session-ssl" className="btn btn-primary btn-block">Using SSL</Link>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    );
};

export default HomePage;
